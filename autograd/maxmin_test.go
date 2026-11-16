package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestMaxMinValues(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		av := []float32{-3, -1, 0, 1, 3, 2, -2, 5}
		bv := []float32{1, -2, 0, 4, -3, 2, 7, -5}
		a := conv2dTestTensor(t, av, []int{len(av)}, device, false)
		b := conv2dTestTensor(t, bv, []int{len(bv)}, device, false)
		hi, lo := Max(a, b), Min(a, b)
		defer hi.ReleaseGraph()
		defer lo.ReleaseGraph()
		gotHi, gotLo := conv2dHost(t, hi, false), conv2dHost(t, lo, false)
		for i := range av {
			if want := float32(math.Max(float64(av[i]), float64(bv[i]))); gotHi[i] != want {
				t.Fatalf("Max[%d] = %g, want %g", i, gotHi[i], want)
			}
			if want := float32(math.Min(float64(av[i]), float64(bv[i]))); gotLo[i] != want {
				t.Fatalf("Min[%d] = %g, want %g", i, gotLo[i], want)
			}
		}
	})
}

func TestMaxMinGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		av := []float32{-3, -1, 0, 1, 3, 2, -2, 5}
		bv := []float32{1, -2, 0, 4, -3, 2, 7, -5}
		for _, tc := range []struct {
			name string
			op   func(*Tensor, *Tensor) *Tensor
			// leftWins reports whether the gradient belongs to a when a<b.
			leftWins bool
		}{
			{"max", Max, false},
			{"min", Min, true},
		} {
			a := conv2dTestTensor(t, av, []int{len(av)}, device, true)
			b := conv2dTestTensor(t, bv, []int{len(bv)}, device, true)
			loss := Sum(tc.op(a, b), 0)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			ga, gb := conv2dHost(t, a, true), conv2dHost(t, b, true)
			for i := range av {
				if ga[i]+gb[i] != 1 {
					t.Fatalf("%s index %d: gradA %g + gradB %g = %g, want 1", tc.name, i, ga[i], gb[i], ga[i]+gb[i])
				}
				wantA := float32(0)
				if (av[i] < bv[i]) == tc.leftWins && av[i] != bv[i] {
					wantA = 1
				}
				if ga[i] != wantA {
					t.Fatalf("%s index %d (a=%g b=%g): gradA %g, want %g", tc.name, i, av[i], bv[i], ga[i], wantA)
				}
			}
			loss.ReleaseGraph()
		}
	})
}

func TestMaxMinTiesGoRight(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4}
		a := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		b := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		loss := Add(Sum(Max(a, b), 0), Sum(Min(a, b), 0))
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		ga, gb := conv2dHost(t, a, true), conv2dHost(t, b, true)
		for i := range values {
			if ga[i] != 0 || gb[i] != 2 {
				t.Fatalf("index %d: tie should send both gradients right, got gradA %g gradB %g", i, ga[i], gb[i])
			}
		}
	})
}

func TestMaxMinBroadcast(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		av := []float32{1, -5, 3, -2, 4, 0}
		bv := []float32{0, 2, -1}
		a := conv2dTestTensor(t, av, []int{2, 3}, device, true)
		b := conv2dTestTensor(t, bv, []int{3}, device, true)
		loss := Sum(Max(a, b), 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		ga, gb := conv2dHost(t, a, true), conv2dHost(t, b, true)
		for i := range av {
			j := i % 3
			if av[i] > bv[j] {
				if ga[i] != 1 {
					t.Fatalf("index %d: a>b but gradA %g", i, ga[i])
				}
			} else if ga[i] != 0 {
				t.Fatalf("index %d: a<=b but gradA %g", i, ga[i])
			}
		}
		for j := range bv {
			var want float32
			for i := 0; i < 2; i++ {
				if av[i*3+j] <= bv[j] {
					want++
				}
			}
			if gb[j] != want {
				t.Fatalf("bias gradient[%d] = %g, want %g", j, gb[j], want)
			}
		}
	})
}

func TestMaxMinNonContiguous(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		av := []float32{-2, -1, 0, 1, 2, 3}
		bv := []float32{0, 0, 0, 0, 0, 0}
		a := conv2dTestTensor(t, av, []int{2, 3}, device, true)
		b := conv2dTestTensor(t, bv, []int{3, 2}, device, false)
		view := Transpose(a, 0, 1)
		if view.IsContiguous() {
			t.Fatal("expected non-contiguous input")
		}
		y := Max(view, b)
		got := conv2dHost(t, y, false)
		order := []int{0, 3, 1, 4, 2, 5}
		for i, j := range order {
			want := float32(math.Max(float64(av[j]), 0))
			if got[i] != want {
				t.Fatalf("output[%d] = %g, want %g", i, got[i], want)
			}
		}
		loss := Sum(y, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, a, true)
		for i, v := range av {
			want := float32(0)
			if v > 0 {
				want = 1
			}
			if grad[i] != want {
				t.Fatalf("gradient[%d] = %g, want %g", i, grad[i], want)
			}
		}
	})
}
