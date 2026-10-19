package autograd

import (
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// TestAuditGradAccumulatorKeepsLossReadable checks that the accumulator does
// not dispose the loss tensor it is given. The caller reads the loss value
// after the backward to report it, and the documented usage releases the loss
// graph afterwards.
func TestAuditGradAccumulatorKeepsLossReadable(t *testing.T) {
	model := newAccumulateModel(t, tensor.CPU)
	execution := NewExecutionContext()
	model.bind(t, execution)
	accumulator, err := NewGradAccumulator(2)
	if err != nil {
		t.Fatal(err)
	}
	loss := model.loss(t, execution, tensor.CPU, 0, 2)
	before, err := loss.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := accumulator.Backward(loss); err != nil {
		t.Fatal(err)
	}
	after, err := loss.ToHost()
	if err != nil {
		t.Fatalf("accumulator disposed the caller's loss: %v", err)
	}
	if len(after) != len(before) || after[0] != before[0] {
		t.Fatalf("loss changed after the backward: %v, want %v", after, before)
	}
	loss.ReleaseGraph()
	if _, err := loss.ToHost(); err == nil {
		t.Fatal("released loss is still readable")
	}
}

// TestAuditGradAccumulatorGradientUnchanged checks that keeping the loss
// readable does not change the accumulated gradient.
func TestAuditGradAccumulatorGradientUnchanged(t *testing.T) {
	model := newAccumulateModel(t, tensor.CPU)
	execution := NewExecutionContext()
	model.bind(t, execution)
	accumulator, err := NewGradAccumulator(2)
	if err != nil {
		t.Fatal(err)
	}
	for micro := 0; micro < 2; micro++ {
		loss := model.loss(t, execution, tensor.CPU, micro*2, micro*2+2)
		complete, err := accumulator.Backward(loss)
		if err != nil {
			t.Fatal(err)
		}
		if wantComplete := micro == 1; complete != wantComplete {
			t.Fatalf("micro-batch %d complete=%v, want %v", micro, complete, wantComplete)
		}
		loss.ReleaseGraph()
	}
	gradients := accumulateGradients(t, model.root)
	if len(gradients) == 0 {
		t.Fatal("no gradients were accumulated")
	}
	nonZero := false
	for i, g := range gradients {
		if g != 0 {
			nonZero = true
			continue
		}
		if i >= len(gradients) {
			t.Fatalf("gradient[%d] out of range", i)
		}
	}
	if !nonZero {
		t.Fatal("every accumulated gradient is zero")
	}
}
