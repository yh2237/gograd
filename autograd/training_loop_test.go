package autograd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"sync"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// loopBatch is one deterministic micro-batch of a fake reader.
type loopBatch struct {
	values []float32
}

// loopReader delivers a fixed list of batches, then reports the end of an epoch.
type loopReader struct {
	mu        sync.Mutex
	batches   []loopBatch
	cursor    int
	epochs    int
	failAfter int
}

func (r *loopReader) Next(ctx context.Context) (loopBatch, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return loopBatch{}, err
	}
	if r.failAfter > 0 && r.cursor == r.failAfter {
		return loopBatch{}, errLoopRead
	}
	if r.cursor == len(r.batches) {
		return loopBatch{}, io.EOF
	}
	batch := r.batches[r.cursor]
	r.cursor++
	return batch, nil
}
func (r *loopReader) NextEpoch() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.epochs++
	r.cursor = 0
	r.failAfter = 0
	return nil
}
func (r *loopReader) delivered() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cursor
}

var errLoopRead = errors.New("loop reader failed")

// loopSampler reports the reader's delivered position, so a savepoint names the
// first micro-batch of the group it was taken for.
type loopSampler struct {
	reader *loopReader
	mu     sync.Mutex
	saved  []byte
	saves  int
	loads  int
}

func (s *loopSampler) MarshalBinary() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.saves++
	s.saved = []byte(fmt.Sprintf("position=%d", s.reader.delivered()))
	return append([]byte(nil), s.saved...), nil
}
func (s *loopSampler) UnmarshalBinary(payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loads++
	s.saved = append([]byte(nil), payload...)
	return nil
}

type loopModel struct {
	module       *Module
	hidden, head *LinearLayer
}

func newLoopModel(t *testing.T, device tensor.Device, seed int64) *loopModel {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	hidden, err := NewLinearLayer(3, 4, device, rng)
	if err != nil {
		t.Fatal(err)
	}
	head, err := NewLinearLayer(4, 1, device, rng)
	if err != nil {
		t.Fatal(err)
	}
	module := &Module{Children: []NamedModule{
		{Name: "hidden", Module: hidden.StateModule()},
		{Name: "head", Module: head.StateModule()},
	}}
	for _, p := range module.NamedParameters() {
		t.Cleanup(p.Value.Close)
	}
	return &loopModel{module: module, hidden: hidden, head: head}
}
func (m *loopModel) forward(x *Tensor) *Tensor {
	return m.head.Forward(ReLU(m.hidden.Forward(x)))
}
func (m *loopModel) gradients(t *testing.T) map[string][]float32 {
	t.Helper()
	out := map[string][]float32{}
	for _, p := range m.module.NamedParameters() {
		grad, err := p.Value.GradToHost()
		if err != nil {
			t.Fatal(err)
		}
		out[p.Name] = grad
	}
	return out
}
func (m *loopModel) assertNoGradients(t *testing.T) {
	t.Helper()
	for name, grad := range m.gradients(t) {
		for _, g := range grad {
			if g != 0 {
				t.Fatalf("%s kept a gradient %g", name, g)
			}
		}
	}
}

func loopBatches(count int) []loopBatch {
	batches := make([]loopBatch, count)
	rng := rand.New(rand.NewSource(5))
	for i := range batches {
		values := make([]float32, 2*3)
		for j := range values {
			values[j] = float32(rng.NormFloat64())
		}
		batches[i] = loopBatch{values: values}
	}
	return batches
}

// loopStep reads one micro-batch, builds a loss and back-propagates it through
// the loop's accumulator. It owns the tensors it creates.
func loopStep(t *testing.T, execution *ExecutionContext, model *loopModel, device tensor.Device) func(context.Context, loopBatch, *GradAccumulator) (float64, bool, error) {
	t.Helper()
	return func(ctx context.Context, batch loopBatch, accumulator *GradAccumulator) (float64, bool, error) {
		if err := ctx.Err(); err != nil {
			return 0, false, err
		}
		x, err := execution.New(batch.values, []int{2, 3}, device, false)
		if err != nil {
			return 0, false, err
		}
		defer x.Close()
		loss := MSELoss(model.forward(x), x, LossOptions{})
		values, err := loss.ToHost()
		if err != nil {
			loss.ReleaseGraph()
			return 0, false, err
		}
		complete, err := accumulator.Backward(loss)
		loss.ReleaseGraph()
		if err != nil {
			return 0, false, err
		}
		return float64(values[0]), complete, nil
	}
}

func TestTrainingLoopAppliesUpdatesAndReports(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		pool2dContext(t, device)
		model := newLoopModel(t, device, 3)
		initial := model.module.StateDict()
		execution := NewExecutionContext()
		if err := execution.BindModule(model.module); err != nil {
			t.Fatal(err)
		}
		optimizer := NewAdamW(model.module.NamedParameters(), .01, 0)
		t.Cleanup(optimizer.Close)
		schedule := NewOneCycle(.01, 6, .5)
		reader := &loopReader{batches: loopBatches(3)}
		var reported []int
		var losses []float64
		loop := TrainingLoop[loopBatch]{
			Reader: reader, Sampler: &loopSampler{reader: reader}, Optimizer: optimizer, Schedule: schedule,
			Updates: 6, Accumulate: 2, LogEvery: 2,
			Step:   loopStep(t, execution, model, device),
			Report: func(update int, loss float64) { reported = append(reported, update); losses = append(losses, loss) },
			Save: func(savepoint []byte) error {
				if savepoint != nil {
					t.Errorf("a completed run saved the savepoint %q", savepoint)
				}
				return nil
			},
		}
		// Three batches per epoch supply twelve micro-batches for six updates.
		if err := loop.Run(context.Background()); err != nil {
			t.Fatal(err)
		}
		if loop.Applied != 6 || reader.epochs != 3 {
			t.Fatalf("applied=%d epoch rolls=%d", loop.Applied, reader.epochs)
		}
		if schedule.Updates() != 6 {
			t.Fatalf("schedule applied %d updates", schedule.Updates())
		}
		if len(reported) != 4 || reported[0] != 1 || reported[1] != 2 || reported[2] != 4 || reported[3] != 6 {
			t.Fatalf("reported %v, want the first, every second and the last update", reported)
		}
		for _, loss := range losses {
			if loss <= 0 {
				t.Fatalf("reported loss %g", loss)
			}
		}
		// A completed run leaves the last group's gradients in place; they are
		// only cleared when another update starts.
		for name, grad := range model.gradients(t) {
			zero := true
			for _, g := range grad {
				zero = zero && g == 0
			}
			if zero {
				t.Fatalf("%s has no gradient after a completed run", name)
			}
		}
		// Every parameter moved, so the optimizer really stepped.
		for name, values := range model.module.StateDict() {
			changed := false
			for i, v := range values {
				if v != initial[name][i] {
					changed = true
				}
			}
			if !changed {
				t.Fatalf("%s did not change", name)
			}
		}
	})
}

func TestTrainingLoopValidation(t *testing.T) {
	model := newLoopModel(t, tensor.CPU, 1)
	optimizer := NewAdamW(model.module.NamedParameters(), .01, 0)
	t.Cleanup(optimizer.Close)
	base := TrainingLoop[loopBatch]{
		Reader: &loopReader{batches: loopBatches(2)}, Sampler: &loopSampler{reader: &loopReader{}}, Optimizer: optimizer,
		Step: func(context.Context, loopBatch, *GradAccumulator) (float64, bool, error) { return 0, true, nil },
	}
	for name, mutate := range map[string]func(*TrainingLoop[loopBatch]){
		"reader":    func(l *TrainingLoop[loopBatch]) { l.Reader = nil },
		"optimizer": func(l *TrainingLoop[loopBatch]) { l.Optimizer = nil },
		"step":      func(l *TrainingLoop[loopBatch]) { l.Step = nil },
		"updates":   func(l *TrainingLoop[loopBatch]) { l.Updates = -1 },
		"applied":   func(l *TrainingLoop[loopBatch]) { l.Applied = 3 },
		"sampler":   func(l *TrainingLoop[loopBatch]) { l.Accumulate = 2; l.Sampler = nil },
	} {
		loop := base
		loop.Updates, loop.Applied = 2, 0
		mutate(&loop)
		if err := loop.Run(context.Background()); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	// A run whose updates are all applied reads nothing and reports nothing.
	reported := 0
	finished := base
	finished.Updates, finished.Applied = 2, 2
	finished.Report = func(int, float64) { reported++ }
	if err := finished.Run(context.Background()); err != nil || reported != 0 {
		t.Fatalf("finished run err=%v reports=%d", err, reported)
	}
}

func TestTrainingLoopInterruption(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		// One micro-batch, a group boundary and the last group cover the
		// savepoint arithmetic.
		for _, tc := range []struct{ groups, after int }{{1, 1}, {3, 4}, {3, 6}} {
			groups, after := tc.groups, tc.after
			t.Run(fmt.Sprintf("groups%d/micro%d", groups, after), func(t *testing.T) {
				pool2dContext(t, device)
				model := newLoopModel(t, device, 7)
				execution := NewExecutionContext()
				if err := execution.BindModule(model.module); err != nil {
					t.Fatal(err)
				}
				optimizer := NewAdamW(model.module.NamedParameters(), .01, 0)
				t.Cleanup(optimizer.Close)
				reader := &loopReader{batches: loopBatches(8)}
				sampler := &loopSampler{reader: reader}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				step := loopStep(t, execution, model, device)
				micro := 0
				var saved []byte
				saves := 0
				loop := TrainingLoop[loopBatch]{
					Reader: reader, Sampler: sampler, Optimizer: optimizer, Updates: 8,
					Accumulate: groups,
					Step: func(ctx context.Context, batch loopBatch, accumulator *GradAccumulator) (float64, bool, error) {
						loss, complete, err := step(ctx, batch, accumulator)
						if err != nil {
							return 0, false, err
						}
						micro++
						if micro == after {
							cancel()
						}
						return loss, complete, nil
					},
					Save: func(savepoint []byte) error {
						saved = append([]byte(nil), savepoint...)
						saves++
						return nil
					},
				}
				if err := loop.Run(ctx); !errors.Is(err, context.Canceled) {
					t.Fatalf("run err=%v", err)
				}
				if saves != 1 {
					t.Fatalf("Save ran %d times", saves)
				}
				// The update count only counts completed groups.
				if loop.Applied != after/groups {
					t.Fatalf("applied=%d, want %d", loop.Applied, after/groups)
				}
				wantSavepoint := fmt.Sprintf("position=%d", (after-1)/groups*groups)
				if after%groups == 0 {
					wantSavepoint = ""
				}
				if string(saved) != wantSavepoint {
					t.Fatalf("savepoint %q, want %q", saved, wantSavepoint)
				}
				model.assertNoGradients(t)
			})
		}
	})
}

func TestTrainingLoopReadErrorSkipsSave(t *testing.T) {
	model := newLoopModel(t, tensor.CPU, 9)
	execution := NewExecutionContext()
	if err := execution.BindModule(model.module); err != nil {
		t.Fatal(err)
	}
	optimizer := NewAdamW(model.module.NamedParameters(), .01, 0)
	t.Cleanup(optimizer.Close)
	reader := &loopReader{batches: loopBatches(4), failAfter: 2}
	loop := TrainingLoop[loopBatch]{
		Reader: reader, Sampler: &loopSampler{reader: reader}, Optimizer: optimizer, Updates: 4, Accumulate: 2,
		Step: loopStep(t, execution, model, tensor.CPU),
		Save: func([]byte) error { return errors.New("Save must not run for a read failure") },
	}
	if err := loop.Run(context.Background()); !errors.Is(err, errLoopRead) {
		t.Fatalf("run err=%v", err)
	}
	if loop.Applied != 1 {
		t.Fatalf("applied=%d, want 1", loop.Applied)
	}
}
