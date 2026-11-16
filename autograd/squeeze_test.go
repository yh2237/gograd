package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestUnsqueezeShape(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4, 5, 6}
		x := conv2dTestTensor(t, values, []int{2, 3}, device, false)
		for _, tc := range []struct {
			axis int
			want []int
		}{
			{0, []int{1, 2, 3}},
			{1, []int{2, 1, 3}},
			{2, []int{2, 3, 1}},
		} {
			y := Unsqueeze(x, tc.axis)
			if !slices.Equal(y.Shape, tc.want) {
				t.Fatalf("Unsqueeze(axis=%d) shape = %v, want %v", tc.axis, y.Shape, tc.want)
			}
			got := conv2dHost(t, y, false)
			if !slices.Equal(got, values) {
				t.Fatalf("Unsqueeze(axis=%d) values = %v, want %v", tc.axis, got, values)
			}
			y.ReleaseGraph()
		}
	})
}

func TestUnsqueezeRejectsAxis(t *testing.T) {
	x := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CPU, false)
	for _, axis := range []int{-1, 2} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("Unsqueeze(axis=%d) did not panic", axis)
				}
			}()
			Unsqueeze(x, axis)
		}()
	}
}

func TestSqueezeShape(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3}
		for _, tc := range []struct {
			shape  []int
			values []float32
			want   []int
		}{
			{[]int{1, 3, 1}, values, []int{3}},
			{[]int{1, 1, 3}, values, []int{3}},
			{[]int{3}, values, []int{3}},
			{[]int{1, 1, 1}, []float32{7}, []int{}},
		} {
			x := conv2dTestTensor(t, tc.values, tc.shape, device, false)
			y := Squeeze(x)
			if !slices.Equal(y.Shape, tc.want) {
				t.Fatalf("Squeeze(%v) shape = %v, want %v", tc.shape, y.Shape, tc.want)
			}
			got := conv2dHost(t, y, false)
			if !slices.Equal(got, tc.values) {
				t.Fatalf("Squeeze(%v) values = %v, want %v", tc.shape, got, tc.values)
			}
			y.ReleaseGraph()
		}
	})
}

func TestSqueezeUnsqueezeGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4}
		x := conv2dTestTensor(t, values, []int{2, 2}, device, true)
		y := Squeeze(Unsqueeze(x, 1))
		if !slices.Equal(y.Shape, []int{2, 2}) {
			t.Fatalf("round trip shape = %v, want [2 2]", y.Shape)
		}
		loss := Sum(y, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		for i := range values {
			if grad[i] != 1 {
				t.Fatalf("gradient[%d] = %g, want 1", i, grad[i])
			}
		}
	})
}
