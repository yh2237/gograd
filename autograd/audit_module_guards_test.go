package autograd

import (
	"context"
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// A run shorter than the warmup span must still start at the documented
// initial rate rather than near the peak.
func TestAuditOneCycleShortRunStartsAtInitialRate(t *testing.T) {
	for _, tc := range []struct {
		total    int
		pctStart float64
	}{
		{5, .1}, {3, .3}, {2, .3},
	} {
		c := NewOneCycle(.01, tc.total, tc.pctStart)
		want := .01 / 25
		if got := c.LR(); math.Abs(got-want) > 1e-12 {
			t.Fatalf("total=%d pctStart=%g: LR() = %g, want %g", tc.total, tc.pctStart, got, want)
		}
	}
}

// Parameters and buffers share one namespace, so a collision must be reported
// rather than silently dropping a tensor.
func TestAuditStateDictRejectsDuplicateName(t *testing.T) {
	m := &Module{
		Parameters: []Parameter{auditParam(t, "x", 1, 2)},
		Buffers:    []Parameter{auditParam(t, "x", 9, 9)},
	}
	defer func() {
		if recover() == nil {
			t.Fatal("StateDict accepted a duplicate state name")
		}
	}()
	m.StateDict()
}

// The nil guard must also cover a typed-nil Trainer.
func TestAuditSetTrainingIgnoresTypedNil(t *testing.T) {
	var m *Module
	SetTraining(false, m)
}

// A resumed run reports its own first update, not the absolute first.
func TestAuditTrainingLoopReportsResumedFirstUpdate(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		pool2dContext(t, device)
		model := newLoopModel(t, device, 3)
		execution := NewExecutionContext()
		if err := execution.BindModule(model.module); err != nil {
			t.Fatal(err)
		}
		optimizer := NewAdamW(model.module.NamedParameters(), .01, 0)
		t.Cleanup(optimizer.Close)
		reader := &loopReader{batches: loopBatches(3)}
		var reported []int
		loop := TrainingLoop[loopBatch]{
			Reader: reader, Sampler: &loopSampler{reader: reader}, Optimizer: optimizer,
			Updates: 10, Applied: 5, LogEvery: 100,
			Step:   loopStep(t, execution, model, device),
			Report: func(update int, loss float64) { reported = append(reported, update) },
		}
		if err := loop.Run(context.Background()); err != nil {
			t.Fatal(err)
		}
		if len(reported) != 2 || reported[0] != 6 || reported[1] != 10 {
			t.Fatalf("reported %v, want [6 10]", reported)
		}
	})
}
