package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestInferenceGPUShapeOps(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		x, err := New(data, []int{2, 3, 4}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		for _, tc := range []struct {
			name string
			op   func() *Tensor
			want []int
		}{
			{"unsqueeze0", func() *Tensor { return Unsqueeze(x, 0) }, []int{1, 2, 3, 4}},
			{"unsqueeze1", func() *Tensor { return Unsqueeze(x, 1) }, []int{2, 1, 3, 4}},
			{"unsqueeze3", func() *Tensor { return Unsqueeze(x, 3) }, []int{2, 3, 4, 1}},
			{"squeeze", func() *Tensor { return Squeeze(Unsqueeze(x, 1)) }, []int{2, 3, 4}},
			{"flatten", func() *Tensor { return Flatten(x, 1, 2) }, []int{2, 12}},
			{"flattenAll", func() *Tensor { return Flatten(x, 0, 2) }, []int{24}},
		} {
			y := tc.op()
			if !slices.Equal(y.Shape, tc.want) {
				t.Fatalf("%s shape = %v, want %v", tc.name, y.Shape, tc.want)
			}
			host, err := y.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, y.Numel())
			for i := range want {
				want[i] = data[i%len(data)]
			}
			conv2dClose(t, tc.name, host, want, 0, 0)
			y.ReleaseGraph()
		}
	})
}
