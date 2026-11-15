package nn

import (
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// A nil gradient must be reported, not dereferenced.
func TestAuditConv1dBackwardRejectsNilGradient(t *testing.T) {
	conv, err := NewConv1dOn(tensor.CPU, 2, 3, 3, 1, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer conv.Close()
	x, err := tensor.FromHostOn(tensor.CPU, []int{1, 3, 2}, make([]float32, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err := conv.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	if _, err := conv.Backward(nil, nil); err == nil {
		t.Fatal("Conv1d.Backward accepted a nil gradient")
	}
}

// A gradient of another shape must be rejected before the backward indexes it.
func TestAuditConv1dBackwardRejectsShapeMismatch(t *testing.T) {
	conv, err := NewConv1dOn(tensor.CPU, 2, 3, 3, 1, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer conv.Close()
	x, err := tensor.FromHostOn(tensor.CPU, []int{1, 3, 2}, make([]float32, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err := conv.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	grad, err := tensor.ZerosOn(tensor.CPU, 1, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer grad.Close()
	if _, err := conv.Backward(nil, grad); err == nil {
		t.Fatal("Conv1d.Backward accepted a gradient with the wrong shape")
	}
}

// A CUDA gradient after a CPU forward must not reach the CUDA path, which has
// no device columns to read.
func TestAuditConv1dBackwardRejectsDeviceMismatch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	conv, err := NewConv1dOn(tensor.CPU, 2, 3, 3, 1, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer conv.Close()
	x, err := tensor.FromHostOn(tensor.CPU, []int{1, 3, 2}, make([]float32, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err := conv.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	grad, err := tensor.ZerosOn(tensor.CUDA, 1, 3, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer grad.Close()
	if _, err := conv.Backward(nil, grad); err == nil {
		t.Fatal("Conv1d.Backward accepted a gradient on another device")
	}
}

// The activation backward sizes its output from the saved input, so a gradient
// of another shape must be rejected instead of indexing past its end.
func TestAuditActivationBackwardRejectsShapeMismatch(t *testing.T) {
	x, err := tensor.FromHostOn(tensor.CPU, []int{2, 3}, make([]float32, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	relu := &ReLU{}
	if _, err := relu.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	short, err := tensor.ZerosOn(tensor.CPU, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer short.Close()
	if _, err := relu.Backward(nil, short); err == nil {
		t.Fatal("activation backward accepted a short gradient")
	}
	if _, err := relu.Backward(nil, nil); err == nil {
		t.Fatal("activation backward accepted a nil gradient")
	}
}
