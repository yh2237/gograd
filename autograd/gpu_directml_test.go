package autograd

import (
	"os"
	"testing"

	"github.com/yh2237/gograd/directml"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func TestDirectMLTensorInference(t *testing.T) {
	if !directml.Available() {
		if os.Getenv("GOGRAD_REQUIRE_DIRECTML") == "1" {
			t.Fatal(directml.Err())
		}
		t.Skipf("DirectML unavailable: %v", directml.Err())
	}
	before := directml.MemoryStats()
	c := NewExecutionContext()
	x, err := c.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, tensor.DirectML, true)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	w, err := c.New([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{3, 4}, tensor.DirectML, true)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	bias, err := c.New([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 4}, tensor.DirectML, true)
	if err != nil {
		t.Fatal(err)
	}
	defer bias.Close()
	assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { MatMul(x, w) })
	var y *Tensor
	c.NoGrad(func() { y = Reshape(MulScalar(Add(MatMul(x, w), bias), .5), 8) })
	defer y.Close()
	if y.RequiresGrad || y.ExecutionContext() != c {
		t.Fatal("inference context/NoGrad lost")
	}
	y.RetainData()
	x.Close()
	w.Close()
	bias.Close()
	y.ReleaseGraph()
	host, err := y.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "DirectML chain", host, []float32{19.5, 23, 26.5, 30, 44, 52, 60, 68}, 1e-5, 1e-5)
	copy := y.Detach()
	defer copy.Close()
	y.Close()
	if err := copy.CopyFrom([]float32{8, 7, 6, 5, 4, 3, 2, 1}); err != nil {
		t.Fatal(err)
	}
	host, err = copy.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "DirectML detached", host, []float32{8, 7, 6, 5, 4, 3, 2, 1}, 0, 0)
	copy.Close()
	if after := directml.MemoryStats(); after.LiveBuffers != before.LiveBuffers || after.LiveBytes != before.LiveBytes {
		t.Fatalf("DirectML tensor leaked: %+v -> %+v", before, after)
	}
}
