package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// accumulateModel is an identical two-layer model with fixed data.
type accumulateModel struct {
	root         *Module
	hidden, head *LinearLayer
	inputs       [][]float32
	targets      [][]float32
}

func newAccumulateModel(t *testing.T, device tensor.Device) *accumulateModel {
	t.Helper()
	rng := rand.New(rand.NewSource(43))
	hidden, err := NewLinearLayer(4, 3, device, rng)
	if err != nil {
		t.Fatal(err)
	}
	head, err := NewLinearLayer(3, 2, device, rng)
	if err != nil {
		t.Fatal(err)
	}
	root := &Module{Children: []NamedModule{
		{Name: "hidden", Module: hidden.StateModule()},
		{Name: "head", Module: head.StateModule()},
	}}
	for _, p := range root.NamedParameters() {
		t.Cleanup(p.Value.Close)
	}
	model := &accumulateModel{root: root, hidden: hidden, head: head}
	for i := 0; i < 8; i++ {
		model.inputs = append(model.inputs, []float32{float32(i%5) / 4, float32(i%3) - 1, .25, -.5})
		model.targets = append(model.targets, []float32{float32(i%2) * .5, .75})
	}
	return model
}

func (m *accumulateModel) bind(t *testing.T, execution *ExecutionContext) {
	t.Helper()
	if err := execution.BindModule(m.root); err != nil {
		t.Fatal(err)
	}
}

// loss builds and returns the loss of the given sample range. The caller owns
// the returned graph.
func (m *accumulateModel) loss(t *testing.T, execution *ExecutionContext, device tensor.Device, from, to int) *Tensor {
	t.Helper()
	values := make([]float32, 0, (to-from)*4)
	targets := make([]float32, 0, (to-from)*2)
	for i := from; i < to; i++ {
		values = append(values, m.inputs[i]...)
		targets = append(targets, m.targets[i]...)
	}
	x, err := execution.New(values, []int{to - from, 4}, device, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(x.Close)
	y, err := execution.New(targets, []int{to - from, 2}, device, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(y.Close)
	return MSELoss(m.head.Forward(ReLU(m.hidden.Forward(x))), y, LossOptions{})
}

func accumulateGradients(t *testing.T, root *Module) []float32 {
	t.Helper()
	var out []float32
	for _, p := range root.NamedParameters() {
		grad, err := p.Value.GradToHost()
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, grad...)
	}
	return out
}

func (m *accumulateModel) clearGradients() {
	for _, p := range m.root.NamedParameters() {
		p.Value.ZeroGrad()
	}
}

// TestGradAccumulatorMatchesSingleBatch splits one batch of eight into four
// micro-batches and checks that the accumulated gradients equal the gradients of
// the single batch, on CPU and on CUDA.
func TestGradAccumulatorMatchesSingleBatch(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		pool2dContext(t, device)
		whole := newAccumulateModel(t, device)
		executionWhole := NewExecutionContext()
		whole.bind(t, executionWhole)
		loss := whole.loss(t, executionWhole, device, 0, 8)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		loss.ReleaseGraph()
		want := accumulateGradients(t, whole.root)

		split := newAccumulateModel(t, device)
		executionSplit := NewExecutionContext()
		split.bind(t, executionSplit)
		accumulator, err := NewGradAccumulator(4)
		if err != nil {
			t.Fatal(err)
		}
		for micro := 0; micro < 4; micro++ {
			microLoss := split.loss(t, executionSplit, device, micro*2, micro*2+2)
			complete, err := accumulator.Backward(microLoss)
			if err != nil {
				t.Fatal(err)
			}
			microLoss.ReleaseGraph()
			if wantComplete := micro == 3; complete != wantComplete {
				t.Fatalf("micro-batch %d complete=%v, want %v", micro, complete, wantComplete)
			}
		}
		if accumulator.Pending() != 0 {
			t.Fatalf("accumulator left %d micro-batches pending", accumulator.Pending())
		}
		got := accumulateGradients(t, split.root)
		if len(got) != len(want) {
			t.Fatalf("gradient count %d, want %d", len(got), len(want))
		}
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-5 {
				t.Fatalf("gradient[%d] = %v, single batch gave %v (diff %g)", i, got[i], want[i], diff)
			}
		}
	})
}

// TestGradAccumulatorPartialGroup documents the partial-group contract: the
// remaining micro-batches keep the 1/groups scaling, so stepping early applies a
// smaller effective batch.
func TestGradAccumulatorPartialGroup(t *testing.T) {
	partial := newAccumulateModel(t, tensor.CPU)
	executionPartial := NewExecutionContext()
	partial.bind(t, executionPartial)
	accumulator, err := NewGradAccumulator(2)
	if err != nil {
		t.Fatal(err)
	}
	for micro := 0; micro < 3; micro++ {
		if micro == 2 {
			// The caller would have stepped and zeroed after the completed group.
			partial.clearGradients()
		}
		loss := partial.loss(t, executionPartial, tensor.CPU, micro, micro+1)
		complete, err := accumulator.Backward(loss)
		if err != nil {
			t.Fatal(err)
		}
		loss.ReleaseGraph()
		if wantComplete := micro == 1; complete != wantComplete {
			t.Fatalf("micro-batch %d complete=%v, want %v", micro, complete, wantComplete)
		}
	}
	if accumulator.Pending() != 1 {
		t.Fatalf("pending %d, want 1", accumulator.Pending())
	}
	// A reference that back-propagates only the third micro-batch, scaled by the
	// group size, matches what the accumulator left in the gradient buffers.
	reference := newAccumulateModel(t, tensor.CPU)
	executionReference := NewExecutionContext()
	reference.bind(t, executionReference)
	loss := reference.loss(t, executionReference, tensor.CPU, 2, 3)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	loss.ReleaseGraph()
	want := accumulateGradients(t, reference.root)
	got := accumulateGradients(t, partial.root)
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i]*0.5)); diff > 1e-6 {
			t.Fatalf("gradient[%d] = %v, want %v", i, got[i], want[i]*0.5)
		}
	}
	accumulator.Reset()
	if accumulator.Pending() != 0 {
		t.Fatalf("Reset left %d pending", accumulator.Pending())
	}
}

func TestGradAccumulatorValidation(t *testing.T) {
	for _, groups := range []int{0, -1} {
		if _, err := NewGradAccumulator(groups); err == nil {
			t.Fatalf("groups %d accepted", groups)
		}
	}
	accumulator, err := NewGradAccumulator(2)
	if err != nil {
		t.Fatal(err)
	}
	if accumulator.Groups() != 2 {
		t.Fatalf("groups %d, want 2", accumulator.Groups())
	}
	if _, err := accumulator.Backward(nil); err == nil {
		t.Fatal("nil loss accepted")
	}
	if accumulator.Pending() != 0 {
		t.Fatal("failed Backward changed the pending count")
	}
	var missing *GradAccumulator
	if _, err := missing.Backward(nil); err == nil {
		t.Fatal("nil accumulator accepted")
	}
	if missing.Pending() != 0 || missing.Groups() != 0 {
		t.Fatal("nil accumulator reported state")
	}
	missing.Reset()
}
