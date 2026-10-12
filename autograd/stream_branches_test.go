package autograd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// branchStepRun trains three independent branches of one step and returns the
// parameters followed by their gradients. When split is true each branch's
// forward is enqueued on its own stream through RunBranches, and join names the
// stream that combines the branches and runs backward; join may be nil for the
// default stream.
func branchStepRun(t *testing.T, split bool, streams []*cuda.Stream, join *cuda.Stream) ([]float32, []float32, error) {
	t.Helper()
	execution := NewExecutionContext()
	rng := rand.New(rand.NewSource(29))
	layers := make([]*LinearLayer, 3)
	inputs := [][]float32{
		{.5, -.25, 1.5, .75},
		{-1, .25, .5, -.75},
		{.125, .5, -1.5, 2},
	}
	root := &Module{}
	for i := range layers {
		layer, err := NewLinearLayer(4, 3, tensor.CUDA, rng)
		if err != nil {
			return nil, nil, err
		}
		layers[i] = layer
		root.Children = append(root.Children, NamedModule{Name: fmt.Sprintf("branch%d", i), Module: layer.StateModule()})
	}
	for _, p := range root.NamedParameters() {
		defer p.Value.Close()
	}
	if err := execution.BindModule(root); err != nil {
		return nil, nil, err
	}
	xs := make([]*Tensor, len(inputs))
	for i, values := range inputs {
		x, err := execution.New(values, []int{1, 4}, tensor.CUDA, false)
		if err != nil {
			return nil, nil, err
		}
		defer x.Close()
		xs[i] = x
	}
	target, err := execution.New([]float32{.25, -.75, .5}, []int{1, 3}, tensor.CUDA, false)
	if err != nil {
		return nil, nil, err
	}
	defer target.Close()
	outputs := make([]*Tensor, len(layers))
	if split {
		branches := make([]StreamBranch, len(layers))
		for i := range branches {
			i := i
			branches[i] = StreamBranch{
				Stream: streams[i],
				Run:    func() error { outputs[i] = layers[i].Forward(xs[i]); return nil },
			}
		}
		if err := RunBranches(branches); err != nil {
			return nil, nil, err
		}
	} else {
		for i := range layers {
			outputs[i] = layers[i].Forward(xs[i])
		}
	}
	combine := func() error {
		sum := outputs[0]
		for _, output := range outputs[1:] {
			sum = Add(sum, output)
		}
		difference := Sub(sum, target)
		loss := Mean(Mul(difference, difference), 0, 1)
		if err := loss.Backward(); err != nil {
			loss.ReleaseGraph()
			return err
		}
		loss.ReleaseGraph()
		return nil
	}
	if join == nil {
		if err := combine(); err != nil {
			return nil, nil, err
		}
	} else if err := cuda.WithStream(join, combine); err != nil {
		return nil, nil, err
	}
	var values, gradients []float32
	for _, p := range root.NamedParameters() {
		host, err := p.Value.ToHost()
		if err != nil {
			return nil, nil, err
		}
		values = append(values, host...)
		grad, err := p.Value.GradToHost()
		if err != nil {
			return nil, nil, err
		}
		gradients = append(gradients, grad...)
	}
	return values, gradients, nil
}

// TestRunBranchesJoinsStreams splits one step's three branch forwards onto three
// streams through RunBranches and checks the parameters and gradients against a
// single-stream run, both when the join lands on the default stream and when it
// lands on an explicit one.
func TestRunBranchesJoinsStreams(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	context, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	wantValues, wantGradients, err := branchStepRun(t, false, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s has %d values, want %d", name, len(got), len(want))
		}
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-5 {
				t.Fatalf("%s[%d] = %v, single-stream run gave %v", name, i, got[i], want[i])
			}
		}
	}
	streams := make([]*cuda.Stream, 3)
	for i := range streams {
		stream, err := cuda.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Destroy()
		streams[i] = stream
	}
	gotValues, gotGradients, err := branchStepRun(t, true, streams, nil)
	if err != nil {
		t.Fatal(err)
	}
	compare("parameter", gotValues, wantValues)
	compare("gradient", gotGradients, wantGradients)
	// The join also lands on an explicit stream chosen by the caller.
	join, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer join.Destroy()
	joinValues, joinGradients, err := branchStepRun(t, true, streams, join)
	if err != nil {
		t.Fatal(err)
	}
	compare("joined parameter", joinValues, wantValues)
	compare("joined gradient", joinGradients, wantGradients)
	if cuda.CurrentStream() != nil {
		t.Fatal("stream leaked out of the split step")
	}
}

// TestRunBranchesValidation covers rejected branch descriptions and error
// propagation, including restoring the thread's stream.
func TestRunBranchesValidation(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	context, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	if err := RunBranches(nil); err == nil {
		t.Fatal("empty branch list accepted")
	}
	if err := RunBranches([]StreamBranch{{Run: func() error { return nil }}}); err == nil {
		t.Fatal("branch without a stream accepted")
	}
	if err := RunBranches([]StreamBranch{{Stream: stream}}); err == nil {
		t.Fatal("branch without a function accepted")
	}
	ran := 0
	err = RunBranches([]StreamBranch{
		{Stream: stream, Run: func() error { ran++; return nil }},
		{Stream: stream, Run: func() error { ran++; return fmt.Errorf("branch failed") }},
	})
	if err == nil || ran != 2 {
		t.Fatalf("branch error %v after %d runs", err, ran)
	}
	if cuda.CurrentStream() != nil {
		t.Fatal("failed split left a stream selected")
	}
	// The branches enqueued before the failure are joined into the caller's
	// stream, so a caller that keeps going cannot free their buffers ahead of the
	// work that is still reading them.
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
}
