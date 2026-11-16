package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestFlattenLayer(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := make([]float32, 24)
		for i := range values {
			values[i] = float32(i)
		}
		for _, tc := range []struct {
			shape []int
			want  []int
		}{
			{[]int{2, 3, 4}, []int{2, 12}},
			{[]int{2, 3}, []int{2, 3}},
			{[]int{6}, []int{6}},
			{[]int{}, []int{}},
		} {
			count := 1
			for _, d := range tc.shape {
				count *= d
			}
			x := conv2dTestTensor(t, values[:count], tc.shape, device, false)
			y := FlattenLayer{}.Forward(x)
			if !slices.Equal(y.Shape, tc.want) {
				t.Fatalf("FlattenLayer(%v) shape = %v, want %v", tc.shape, y.Shape, tc.want)
			}
			got := conv2dHost(t, y, false)
			if !slices.Equal(got, values[:count]) {
				t.Fatalf("FlattenLayer(%v) changed the values", tc.shape)
			}
			y.ReleaseGraph()
		}
	})
}

func TestFlattenLayerInSequential(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4, 5, 6}
		x := conv2dTestTensor(t, values, []int{1, 2, 3}, device, true)
		model := Sequential{FlattenLayer{}, GELULayer{}}
		y := model.Forward(x)
		if !slices.Equal(y.Shape, []int{1, 6}) {
			t.Fatalf("shape = %v, want [1 6]", y.Shape)
		}
		loss := Sum(y, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		for i := range values {
			if grad[i] == 0 {
				t.Fatalf("gradient[%d] = 0, want a nonzero value", i)
			}
		}
	})
}
