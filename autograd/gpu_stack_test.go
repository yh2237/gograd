package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestInferenceGPUStack(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		if device == fixtureDevice {
			t.Skip("the fixture backend has no concat")
		}
		data := make([]float32, 6)
		for i := range data {
			data[i] = float32(i)
		}
		a, err := New(data, []int{2, 3}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()
		y := Stack(0, a, a)
		defer y.ReleaseGraph()
		if !slices.Equal(y.Shape, []int{2, 2, 3}) {
			t.Fatalf("shape = %v, want [2 2 3]", y.Shape)
		}
		host, err := y.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		want := append(append([]float32(nil), data...), data...)
		conv2dClose(t, "stack", host, want, 0, 0)
	})
}
