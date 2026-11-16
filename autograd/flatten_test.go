package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestFlattenShape(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := make([]float32, 24)
		for i := range values {
			values[i] = float32(i)
		}
		for _, tc := range []struct {
			shape      []int
			start, end int
			want       []int
		}{
			{[]int{2, 3, 4}, 0, 2, []int{24}},
			{[]int{2, 3, 4}, 1, 2, []int{2, 12}},
			{[]int{2, 3, 4}, 0, 1, []int{6, 4}},
			{[]int{2, 3, 4}, 1, -1, []int{2, 12}},
			{[]int{2, 3, 4}, -2, -1, []int{2, 12}},
			{[]int{2, 3, 4}, 1, 1, []int{2, 3, 4}},
			{[]int{2, 3, 4}, 0, 0, []int{2, 3, 4}},
		} {
			x := conv2dTestTensor(t, values, tc.shape, device, false)
			y := Flatten(x, tc.start, tc.end)
			if !slices.Equal(y.Shape, tc.want) {
				t.Fatalf("Flatten(%v, %d, %d) shape = %v, want %v", tc.shape, tc.start, tc.end, y.Shape, tc.want)
			}
			got := conv2dHost(t, y, false)
			if !slices.Equal(got, values) {
				t.Fatalf("Flatten(%v, %d, %d) changed the values", tc.shape, tc.start, tc.end)
			}
			y.ReleaseGraph()
		}
	})
}

func TestFlattenScalar(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{7}, []int{}, device, false)
		for _, tc := range []struct{ start, end int }{{0, 0}, {0, -1}} {
			y := Flatten(x, tc.start, tc.end)
			if !slices.Equal(y.Shape, []int{1}) {
				t.Fatalf("Flatten(scalar, %d, %d) shape = %v, want [1]", tc.start, tc.end, y.Shape)
			}
			if got := conv2dHost(t, y, false); !slices.Equal(got, []float32{7}) {
				t.Fatalf("Flatten(scalar, %d, %d) values = %v, want [7]", tc.start, tc.end, got)
			}
			y.ReleaseGraph()
		}
	})
}

func TestFlattenRejectsRange(t *testing.T) {
	x := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, tensor.CPU, false)
	for _, tc := range []struct{ start, end int }{
		{-4, 1},
		{0, 2},
		{2, 1},
		{3, 3},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("Flatten(%d, %d) did not panic", tc.start, tc.end)
				}
			}()
			Flatten(x, tc.start, tc.end)
		}()
	}
}

func TestFlattenGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4, 5, 6}
		x := conv2dTestTensor(t, values, []int{2, 3}, device, true)
		y := Flatten(x, 0, 1)
		if !slices.Equal(y.Shape, []int{6}) {
			t.Fatalf("shape = %v, want [6]", y.Shape)
		}
		loss := Sum(y, 0)
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
