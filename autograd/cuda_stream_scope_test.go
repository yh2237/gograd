package autograd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestWithCUDAAppliesAndRestores(t *testing.T) {
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
	execution := NewExecutionContext()
	// An unconfigured context leaves the thread's selection untouched.
	if err := execution.WithCUDA(func() error {
		if cuda.CurrentStream() != nil {
			return fmt.Errorf("unconfigured context changed the stream")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := execution.SetCUDA(0, stream); err != nil {
		t.Fatal(err)
	}
	if device, got := execution.CUDA(); device != 0 || got != stream {
		t.Fatalf("recorded device %d stream %p, want 0 and %p", device, got, stream)
	}
	inside := false
	if err := execution.WithCUDA(func() error {
		inside = true
		if cuda.CurrentStream() != stream {
			return fmt.Errorf("stream not selected inside WithCUDA")
		}
		if device, err := cuda.CurrentDevice(); err != nil || device != 0 {
			return fmt.Errorf("device %d (%v) inside WithCUDA", device, err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !inside {
		t.Fatal("WithCUDA did not run the callback")
	}
	if cuda.CurrentStream() != nil {
		t.Fatal("WithCUDA did not restore the previous stream")
	}
	// The restore also happens when the callback fails.
	failed := false
	if err := execution.WithCUDA(func() error { failed = true; return fmt.Errorf("boom") }); err == nil || !failed {
		t.Fatal("WithCUDA did not report the callback error")
	}
	if cuda.CurrentStream() != nil {
		t.Fatal("WithCUDA did not restore the stream after an error")
	}
	// The nested case restores to the outer stream, not the default.
	outer, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer outer.Destroy()
	if err := cuda.WithStream(outer, func() error {
		return execution.WithCUDA(func() error { return nil })
	}); err != nil {
		t.Fatal(err)
	}
	if cuda.CurrentStream() != nil {
		t.Fatal("nested scopes leaked a stream")
	}
	if err := execution.SetCUDA(99, nil); err == nil {
		t.Fatal("out-of-range device accepted")
	}
	if err := (*ExecutionContext)(nil).SetCUDA(0, nil); err == nil {
		t.Fatal("nil context accepted")
	}
	if err := execution.WithCUDA(nil); err == nil {
		t.Fatal("nil callback accepted")
	}
}

// TestCaptureFollowsImplicitStream checks that cuBLAS follows the thread's
// current stream without an explicit SetBLASStream, which is required for
// capture: work outside the capturing stream would not be recorded.
func TestCaptureFollowsImplicitStream(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	context, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	execution := NewExecutionContext()
	a := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, tensor.CUDA, false)
	b := conv2dTestTensor(t, []float32{1, 0, 0, 1, 1, 1}, []int{3, 2}, tensor.CUDA, false)
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	var out *Tensor
	if err := execution.SetCUDA(0, stream); err != nil {
		t.Fatal(err)
	}
	graph, err := cuda.Capture(stream, func() error {
		return execution.WithCUDA(func() error {
			out = MatMul(a, b)
			return nil
		})
	})
	if err != nil {
		t.Fatalf("capture without SetBLASStream: %v", err)
	}
	defer graph.Close()
	if out == nil {
		t.Fatal("capture did not build a graph")
	}
	defer out.Close()
	// Capture records work instead of running it, so replay before reading.
	if err := graph.Launch(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	values, err := out.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{4, 5, 10, 11}
	if len(values) != len(want) {
		t.Fatalf("captured matmul returned %d values, want %d", len(values), len(want))
	}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("captured matmul[%d] = %v, want %v", i, values[i], want[i])
		}
	}
}

// cudaStreamRun trains one model with the given stream selected through its
// execution context, and returns the parameters along with a finalizer that
// reads them back. Splitting prepare from finalize lets the test enqueue two
// runs on two streams before either readback.
func cudaStreamRun(t *testing.T, seed int64, stream *cuda.Stream, steps int) (func() ([]float32, error), error) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	hidden, err := NewLinearLayer(6, 8, tensor.CUDA, rng)
	if err != nil {
		return nil, err
	}
	head, err := NewLinearLayer(8, 2, tensor.CUDA, rng)
	if err != nil {
		return nil, err
	}
	root := &Module{Children: []NamedModule{
		{Name: "hidden", Module: hidden.StateModule()},
		{Name: "head", Module: head.StateModule()},
	}}
	execution := NewExecutionContext()
	if err := execution.BindModule(root); err != nil {
		return nil, err
	}
	if err := execution.SetCUDA(0, stream); err != nil {
		return nil, err
	}
	inputs := []float32{.5, -.25, 1.5, .75, -.5, .25}
	targets := []float32{.25, -.75}
	var x, target *Tensor
	if err := execution.WithCUDA(func() error {
		var err error
		if x, err = execution.New(inputs, []int{1, 6}, tensor.CUDA, false); err != nil {
			return err
		}
		target, err = execution.New(targets, []int{1, 2}, tensor.CUDA, false)
		return err
	}); err != nil {
		return nil, err
	}
	optimizer := NewAdamW(root.NamedParameters(), .01, .001)
	if err := execution.WithCUDA(func() error {
		for step := 0; step < steps; step++ {
			optimizer.ZeroGrad()
			prediction := head.Forward(ReLU(hidden.Forward(x)))
			difference := Sub(prediction, target)
			loss := Mean(Mul(difference, difference), 0, 1)
			if err := loss.Backward(); err != nil {
				loss.ReleaseGraph()
				return err
			}
			loss.ReleaseGraph()
			optimizer.Step()
		}
		return nil
	}); err != nil {
		optimizer.Close()
		return nil, err
	}
	return func() ([]float32, error) {
		defer optimizer.Close()
		defer x.Close()
		defer target.Close()
		for _, p := range root.NamedParameters() {
			defer p.Value.Close()
		}
		var out []float32
		err := execution.WithCUDA(func() error {
			for _, p := range root.NamedParameters() {
				values, err := p.Value.ToHost()
				if err != nil {
					return err
				}
				out = append(out, values...)
			}
			return nil
		})
		return out, err
	}, nil
}

// splitStepRun trains two independent branches of one model, optionally running
// each branch's forward on its own stream and joining them with an event, and
// returns the parameters followed by their gradients.
func splitStepRun(t *testing.T, left, right *cuda.Stream) ([]float32, []float32, error) {
	t.Helper()
	execution := NewExecutionContext()
	rng := rand.New(rand.NewSource(19))
	first, err := NewLinearLayer(4, 3, tensor.CUDA, rng)
	if err != nil {
		return nil, nil, err
	}
	second, err := NewLinearLayer(4, 3, tensor.CUDA, rng)
	if err != nil {
		return nil, nil, err
	}
	root := &Module{Children: []NamedModule{
		{Name: "first", Module: first.StateModule()},
		{Name: "second", Module: second.StateModule()},
	}}
	for _, p := range root.NamedParameters() {
		defer p.Value.Close()
	}
	if err := execution.BindModule(root); err != nil {
		return nil, nil, err
	}
	inputsA := []float32{.5, -.25, 1.5, .75}
	inputsB := []float32{-1, .25, .5, -.75}
	targets := []float32{.25, -.75, .5}
	xa, err := execution.New(inputsA, []int{1, 4}, tensor.CUDA, false)
	if err != nil {
		return nil, nil, err
	}
	defer xa.Close()
	xb, err := execution.New(inputsB, []int{1, 4}, tensor.CUDA, false)
	if err != nil {
		return nil, nil, err
	}
	defer xb.Close()
	target, err := execution.New(targets, []int{1, 3}, tensor.CUDA, false)
	if err != nil {
		return nil, nil, err
	}
	defer target.Close()
	var ya, yb *Tensor
	if left == nil || right == nil {
		ya = first.Forward(xa)
		yb = second.Forward(xb)
	} else {
		if err := cuda.WithStream(left, func() error { ya = first.Forward(xa); return nil }); err != nil {
			return nil, nil, err
		}
		if err := cuda.WithStream(right, func() error { yb = second.Forward(xb); return nil }); err != nil {
			return nil, nil, err
		}
		// The consumer stream must not read the other branch before it is done.
		event, err := cuda.NewEvent(false)
		if err != nil {
			return nil, nil, err
		}
		defer event.Destroy()
		if err := event.Record(right); err != nil {
			return nil, nil, err
		}
		if err := left.WaitEvent(event); err != nil {
			return nil, nil, err
		}
	}
	// Join, loss and backward run on the caller's stream (or on left when split).
	run := func() error {
		sum := Add(ya, yb)
		difference := Sub(sum, target)
		loss := Mean(Mul(difference, difference), 0, 1)
		if err := loss.Backward(); err != nil {
			loss.ReleaseGraph()
			return err
		}
		loss.ReleaseGraph()
		return nil
	}
	if left == nil {
		if err := run(); err != nil {
			return nil, nil, err
		}
	} else if err := cuda.WithStream(left, run); err != nil {
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

// TestSplitStepAcrossStreams splits one step's two branch forwards onto two
// streams, joins them with an event, and checks the parameters and gradients
// against a single-stream run of the same step.
func TestSplitStepAcrossStreams(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	context, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	wantValues, wantGradients, err := splitStepRun(t, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	left, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer left.Destroy()
	right, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer right.Destroy()
	gotValues, gotGradients, err := splitStepRun(t, left, right)
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
	compare("parameter", gotValues, wantValues)
	compare("gradient", gotGradients, wantGradients)
	if cuda.CurrentStream() != nil {
		t.Fatal("stream leaked out of the split step")
	}
}

// TestTwoStreamsInOneThread interleaves two independent training runs on two
// streams from one OS thread and checks both against runs made on the default
// stream. The stream-keyed pool and the stream-following cuBLAS handle make the
// interleaving safe: before those, reuse crossed streams and changed results.
func TestTwoStreamsInOneThread(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	const steps = 8
	seeds := []int64{11, 23}
	reference := make([][]float32, len(seeds))
	for i, seed := range seeds {
		finalize, err := cudaStreamRun(t, seed, nil, steps)
		if err != nil {
			t.Fatal(err)
		}
		if reference[i], err = finalize(); err != nil {
			t.Fatal(err)
		}
	}
	streams := make([]*cuda.Stream, len(seeds))
	finalize := make([]func() ([]float32, error), len(seeds))
	for i, seed := range seeds {
		stream, err := cuda.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Destroy()
		streams[i] = stream
		if finalize[i], err = cudaStreamRun(t, seed, stream, steps); err != nil {
			t.Fatal(err)
		}
	}
	// Both runs are enqueued on their own streams before either is read back.
	for i := range finalize {
		got, err := finalize[i]()
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		if len(got) != len(reference[i]) {
			t.Fatalf("run %d returned %d values, want %d", i, len(got), len(reference[i]))
		}
		for j := range got {
			if math.IsNaN(float64(got[j])) || math.Abs(float64(got[j]-reference[i][j])) > 1e-5 {
				t.Fatalf("run %d parameter %d = %v, default-stream run gave %v", i, j, got[j], reference[i][j])
			}
		}
	}
	if cuda.CurrentStream() != nil {
		t.Fatal("stream leaked out of the test")
	}
}
