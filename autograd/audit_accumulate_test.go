package autograd

import (
	"math"
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
	// The accumulated gradient must still match a plain backward of the same
	// loss on an identical model.
	twin := newAccumulateModel(t, tensor.CPU)
	twinExecution := NewExecutionContext()
	twin.bind(t, twinExecution)
	twinLoss := twin.loss(t, twinExecution, tensor.CPU, 0, 2)
	if err := twinLoss.Backward(); err != nil {
		t.Fatal(err)
	}
	twinLoss.ReleaseGraph()
	accumulated := accumulateGradients(t, model.root)
	direct := accumulateGradients(t, twin.root)
	if len(accumulated) != len(direct) {
		t.Fatalf("gradient count %d != %d", len(accumulated), len(direct))
	}
	// The accumulator scales every micro-batch by 1/groups.
	for i := range accumulated {
		want := direct[i] / 2
		if diff := math.Abs(float64(accumulated[i] - want)); diff > 1e-5 {
			t.Fatalf("gradient[%d]=%v, direct backward gave %v", i, accumulated[i], want)
		}
	}
	loss.ReleaseGraph()
	if _, err := loss.ToHost(); err == nil {
		t.Fatal("released loss is still readable")
	}
}
