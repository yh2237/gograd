package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestStackShape(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		a := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, device, false)
		b := conv2dTestTensor(t, []float32{5, 6, 7, 8}, []int{2, 2}, device, false)
		for _, tc := range []struct {
			axis int
			want []int
		}{
			{0, []int{2, 2, 2}},
			{1, []int{2, 2, 2}},
			{2, []int{2, 2, 2}},
			{-1, []int{2, 2, 2}},
		} {
			y := Stack(tc.axis, a, b)
			if !slices.Equal(y.Shape, tc.want) {
				t.Fatalf("Stack(axis=%d) shape = %v, want %v", tc.axis, y.Shape, tc.want)
			}
			y.ReleaseGraph()
		}
	})
}

func TestStackValues(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		a := conv2dTestTensor(t, []float32{1, 2}, []int{2}, device, false)
		b := conv2dTestTensor(t, []float32{3, 4}, []int{2}, device, false)
		y := Stack(0, a, b)
		if !slices.Equal(y.Shape, []int{2, 2}) {
			t.Fatalf("shape = %v, want [2 2]", y.Shape)
		}
		got := conv2dHost(t, y, false)
		want := []float32{1, 2, 3, 4}
		if !slices.Equal(got, want) {
			t.Fatalf("values = %v, want %v", got, want)
		}
		y.ReleaseGraph()
	})
}

func TestStackRejectsInvalid(t *testing.T) {
	a := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CPU, false)
	b := conv2dTestTensor(t, []float32{1, 2, 3}, []int{3}, tensor.CPU, false)
	for _, tc := range []struct {
		name  string
		axis  int
		input []*Tensor
	}{
		{"empty", 0, nil},
		{"axis too large", 2, []*Tensor{a, a}},
		{"axis negative", -3, []*Tensor{a, a}},
		{"shape mismatch", 0, []*Tensor{a, b}},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: Stack did not panic", tc.name)
				}
			}()
			Stack(tc.axis, tc.input...)
		}()
	}
}

func TestStackGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		a := conv2dTestTensor(t, []float32{1, 2}, []int{2}, device, true)
		b := conv2dTestTensor(t, []float32{3, 4}, []int{2}, device, true)
		loss := Sum(Stack(0, a, b), 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		for _, tc := range []struct {
			name string
			x    *Tensor
		}{{"a", a}, {"b", b}} {
			grad := conv2dHost(t, tc.x, true)
			for i := range grad {
				if grad[i] != 1 {
					t.Fatalf("%s gradient[%d] = %g, want 1", tc.name, i, grad[i])
				}
			}
		}
	})
}
