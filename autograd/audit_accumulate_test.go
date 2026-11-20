package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestAuditGradAccumulatorKeepsLossReadable(t *testing.T) {
	pool2dDevices(t, testGradAccumulatorKeepsLossReadable)
}

func testGradAccumulatorKeepsLossReadable(t *testing.T, device tensor.Device) {
	model := newAccumulateModel(t, device)
	execution := NewExecutionContext()
	model.bind(t, execution)
	accumulator, err := NewGradAccumulator(2)
	if err != nil {
		t.Fatal(err)
	}
	loss := model.loss(t, execution, device, 0, 2)
	defer loss.ReleaseGraph()
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
	if err := loss.Backward(); err == nil {
		t.Fatal("loss history survived gradient accumulation")
	}
	twin := newAccumulateModel(t, device)
	twinExecution := NewExecutionContext()
	twin.bind(t, twinExecution)
	twinLoss := twin.loss(t, twinExecution, device, 0, 2)
	defer twinLoss.ReleaseGraph()
	if err := twinLoss.Backward(); err != nil {
		t.Fatal(err)
	}
	accumulated := accumulateGradients(t, model.root)
	direct := accumulateGradients(t, twin.root)
	if len(accumulated) != len(direct) {
		t.Fatalf("gradient count %d != %d", len(accumulated), len(direct))
	}
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
