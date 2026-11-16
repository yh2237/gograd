package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// MatMul reads a transposed view of a contiguous weight through SGEMM's transB
// path instead of materializing the transpose. The values and both gradients
// must match the hand-computed product.
func TestAuditMatMulTransposedViewGradients(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		a := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, true)
		defer a.Close()
		w := conv2dTestTensor(t, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, []int{4, 3}, device, true)
		defer w.Close()
		y := MatMul(a, Transpose(w, 0, 1))
		if got, want := conv2dHost(t, y, false), []float32{1, 2, 3, 6, 4, 5, 6, 15}; !slices.Equal(got, want) {
			t.Fatalf("output %v, want %v", got, want)
		}
		loss := Sum(y, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		da := conv2dHost(t, a, true)
		dw := conv2dHost(t, w, true)
		loss.ReleaseGraph()
		for i, v := range da {
			if v != 2 {
				t.Fatalf("input gradient[%d] = %g, want 2", i, v)
			}
		}
		if want := []float32{5, 7, 9, 5, 7, 9, 5, 7, 9, 5, 7, 9}; !slices.Equal(dw, want) {
			t.Fatalf("weight gradient %v, want %v", dw, want)
		}
	})
}

// The batched path must also read a transposed view without materializing it.
// The reference is a direct float64 triple loop over the same operands.
func TestAuditMatMulBatchedTransposedView(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		const n, i, j, d = 2, 3, 5, 4
		av := make([]float32, n*i*d)
		for x := range av {
			av[x] = float32(x%7-3) / 4
		}
		kv := make([]float32, n*j*d)
		for x := range kv {
			kv[x] = float32(x%5-2) / 4
		}
		a := conv2dTestTensor(t, av, []int{n, i, d}, device, true)
		defer a.Close()
		k := conv2dTestTensor(t, kv, []int{n, j, d}, device, true)
		defer k.Close()
		y := MatMul(a, Transpose(k, 1, 2))
		got := conv2dHost(t, y, false)
		want := make([]float32, n*i*j)
		for b := 0; b < n; b++ {
			for r := 0; r < i; r++ {
				for c := 0; c < j; c++ {
					var sum float64
					for e := 0; e < d; e++ {
						sum += float64(av[(b*i+r)*d+e]) * float64(kv[(b*j+c)*d+e])
					}
					want[(b*i+r)*j+c] = float32(sum)
				}
			}
		}
		for x := range want {
			if diff := got[x] - want[x]; diff < -1e-6 || diff > 1e-6 {
				t.Fatalf("output[%d] = %g, want %g", x, got[x], want[x])
			}
		}
		loss := Sum(y, 0, 1, 2)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		da := conv2dHost(t, a, true)
		dk := conv2dHost(t, k, true)
		loss.ReleaseGraph()
		for b := 0; b < n; b++ {
			for r := 0; r < i; r++ {
				for e := 0; e < d; e++ {
					var sum float64
					for c := 0; c < j; c++ {
						sum += float64(kv[(b*j+c)*d+e])
					}
					if diff := float64(da[(b*i+r)*d+e]) - sum; diff < -1e-6 || diff > 1e-6 {
						t.Fatalf("input gradient[%d,%d,%d] = %g, want %g", b, r, e, da[(b*i+r)*d+e], sum)
					}
				}
			}
		}
		for b := 0; b < n; b++ {
			for c := 0; c < j; c++ {
				for e := 0; e < d; e++ {
					var sum float64
					for r := 0; r < i; r++ {
						sum += float64(av[(b*i+r)*d+e])
					}
					if diff := float64(dk[(b*j+c)*d+e]) - sum; diff < -1e-6 || diff > 1e-6 {
						t.Fatalf("weight gradient[%d,%d,%d] = %g, want %g", b, c, e, dk[(b*j+c)*d+e], sum)
					}
				}
			}
		}
	})
}