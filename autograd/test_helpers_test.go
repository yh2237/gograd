package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// These helpers started with Conv2d parity tests and are also shared by the
// execution-context and storage-lifetime tests.
func conv2dClose(t *testing.T, name string, got, want []float32, atol, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d != %d", name, len(got), len(want))
	}
	for i, v := range got {
		diff := math.Abs(float64(v) - float64(want[i]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || diff > atol+rtol*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d]=%.9g want %.9g (diff %.9g)", name, i, v, want[i], diff)
		}
	}
}

func conv2dTestTensor(t *testing.T, v []float32, shape []int, device tensor.Device, grad bool) *Tensor {
	t.Helper()
	x, err := New(v, shape, device, grad)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(x.Close)
	return x
}

func conv2dHost(t *testing.T, x *Tensor, grad bool) []float32 {
	t.Helper()
	var v []float32
	var err error
	if grad {
		v, err = x.GradToHost()
	} else {
		v, err = x.ToHost()
	}
	if err != nil {
		t.Fatal(err)
	}
	return v
}
