package autograd

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// cudaRun trains one small model and returns its flattened parameters. It uses a
// single-sample batch and no gradient clipping so every kernel on this path is
// elementwise or a fixed-shape GEMM: two runs of the same seed are then
// comparable, and any difference between them comes from cross-run interference.
// A nil stream selects the thread's default stream.
func cudaRun(t *testing.T, seed int64, steps int, stream *cuda.Stream) ([]float32, error) {
	t.Helper()
	context, err := NewCUDAContext()
	if err != nil {
		return nil, err
	}
	defer context.Close()
	if stream != nil {
		cuda.SetCurrentStream(stream)
		defer cuda.SetCurrentStream(nil)
	}
	execution := NewExecutionContext()
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
	for _, p := range root.NamedParameters() {
		defer p.Value.Close()
	}
	if err := execution.BindModule(root); err != nil {
		return nil, err
	}
	// Fixed inputs and targets, independent of the seed, keep the two runs comparable.
	inputs := []float32{.5, -.25, 1.5, .75, -.5, .25}
	targets := []float32{.25, -.75}
	x, err := execution.New(inputs, []int{1, 6}, tensor.CUDA, false)
	if err != nil {
		return nil, err
	}
	defer x.Close()
	target, err := execution.New(targets, []int{1, 2}, tensor.CUDA, false)
	if err != nil {
		return nil, err
	}
	defer target.Close()
	optimizer := NewAdamW(root.NamedParameters(), .01, .001)
	defer optimizer.Close()
	for step := 0; step < steps; step++ {
		optimizer.ZeroGrad()
		prediction := head.Forward(ReLU(hidden.Forward(x)))
		difference := Sub(prediction, target)
		loss := Mean(Mul(difference, difference), 0, 1)
		values, err := loss.ToHost()
		if err != nil {
			loss.ReleaseGraph()
			return nil, err
		}
		if math.IsNaN(float64(values[0])) {
			loss.ReleaseGraph()
			return nil, fmt.Errorf("loss became NaN at step %d", step)
		}
		if err := loss.Backward(); err != nil {
			loss.ReleaseGraph()
			return nil, err
		}
		loss.ReleaseGraph()
		optimizer.Step()
	}
	out := make([]float32, 0, 8*6+8+2*8+2)
	for _, p := range root.NamedParameters() {
		values, err := p.Value.ToHost()
		if err != nil {
			return nil, err
		}
		out = append(out, values...)
	}
	return out, nil
}

// runConcurrently starts one locked-thread run per seed and returns the results.
func runConcurrently(t *testing.T, seeds []int64, steps int) ([][]float32, error) {
	t.Helper()
	streams := make([]*cuda.Stream, len(seeds))
	for i := range streams {
		stream, err := cuda.NewStream()
		if err != nil {
			return nil, err
		}
		defer stream.Destroy()
		streams[i] = stream
	}
	results := make([][]float32, len(seeds))
	errs := make([]error, len(seeds))
	var wait sync.WaitGroup
	for i, seed := range seeds {
		wait.Add(1)
		go func(i int, seed int64) {
			defer wait.Done()
			results[i], errs[i] = cudaRun(t, seed, steps, streams[i])
		}(i, seed)
	}
	wait.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

// TestConcurrentCUDATrainingIsolated runs two independent CUDA training runs at
// the same time on separate OS threads, each with its own context, model,
// optimizer and stream, and compares them with the same runs executed one at a
// time. The runs share the process-wide CUDA pool and the kernel caches, so the
// comparison covers the per-thread cuBLAS handle, per-thread streams, the
// allocator and the optimizer state.
func TestConcurrentCUDATrainingIsolated(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	const steps = 12
	seeds := []int64{11, 23}
	reference := make([][]float32, len(seeds))
	for i, seed := range seeds {
		// A separate goroutine per reference run gives each one its own OS thread,
		// so a mounted handle cannot carry state into the next run.
		values, err := runConcurrently(t, []int64{seed}, steps)
		if err != nil {
			t.Fatal(err)
		}
		reference[i] = values[0]
	}
	results, err := runConcurrently(t, seeds, steps)
	if err != nil {
		t.Fatal(err)
	}
	for i := range results {
		if len(results[i]) != len(reference[i]) {
			t.Fatalf("run %d returned %d values, want %d", i, len(results[i]), len(reference[i]))
		}
		for j := range results[i] {
			got, want := results[i][j], reference[i][j]
			if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("run %d parameter %d = %v, isolated run gave %v", i, j, got, want)
			}
		}
	}
}

// TestClipPlanFollowsParameterAddresses checks that the per-thread clip plan is
// rebuilt when parameter addresses change and that it is not shared between OS
// threads.
func TestClipPlanFollowsParameterAddresses(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	first := Parameter{Name: "first", Value: conv2dTestTensor(t, []float32{1, 2, 3}, []int{3}, tensor.CUDA, true)}
	second := Parameter{Name: "second", Value: conv2dTestTensor(t, []float32{4, 5}, []int{2}, tensor.CUDA, true)}
	plan := getClipPlan([]Parameter{first})
	if len(plan.ptrs) != 1 {
		t.Fatalf("plan has %d pointers", len(plan.ptrs))
	}
	// A different parameter set rebuilds the plan.
	other := getClipPlan([]Parameter{second})
	if len(other.ptrs) != 1 || other.ptrs[0] == plan.ptrs[0] {
		t.Fatal("plan did not follow the new parameter addresses")
	}
	// An independently locked OS thread keeps its own clip state. A thread that
	// previously did CUDA work may already have an entry, so compare identities
	// rather than requiring an empty one.
	mineID := cuda.ThreadID()
	mine := clipStateForThread()
	done := make(chan struct{})
	go func() {
		defer close(done)
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if cuda.ThreadID() == mineID {
			return // Same thread id; nothing to distinguish here.
		}
		if currentClipState() == mine {
			t.Error("clip state is shared between OS threads")
		}
	}()
	<-done
}

// TestClipPlanChecksElementCounts covers reuse of a pooled address by a
// parameter with a different element count, where pointer equality alone would
// keep a stale plan.
func TestClipPlanChecksElementCounts(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	// Both shapes round into the same size class, so the pool can reuse the block.
	first := conv2dTestTensor(t, []float32{1, 2, 3}, []int{3}, tensor.CUDA, true)
	plan := getClipPlan([]Parameter{{Name: "first", Value: first}})
	address := plan.ptrs[0]
	first.Close()
	second := conv2dTestTensor(t, []float32{9, 8}, []int{2}, tensor.CUDA, true)
	if second.buf.Pointer() != address {
		t.Skip("pool did not reuse the address; nothing to compare")
	}
	rebuilt := getClipPlan([]Parameter{{Name: "second", Value: second}})
	if len(rebuilt.ptrs) != 1 || rebuilt.ptrs[0] != address {
		t.Fatalf("plan addresses %v, want [%d]", rebuilt.ptrs, address)
	}
	if rebuilt.sizesHost[0] != 2 {
		t.Fatalf("plan kept element count %d, want 2", rebuilt.sizesHost[0])
	}
}

// TestConcurrentClipIsolated clips gradients on two OS threads at the same time
// and checks that each thread reads back its own norm.
func TestConcurrentClipIsolated(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	// The tensors are created on this thread; each worker only uploads gradients,
	// clips and reads the norm back, which is what the per-thread state protects.
	values := [][]float32{{.5, -.25, 1.5}, {3, -4}}
	parameters := make([]Parameter, len(values))
	for i, v := range values {
		parameters[i] = Parameter{Name: fmt.Sprintf("p%d", i), Value: conv2dTestTensor(t, v, []int{len(v)}, tensor.CUDA, true)}
	}
	ctx.Close()
	want := make([]float64, len(values))
	for i, v := range values {
		var sum float64
		for _, value := range v {
			sum += float64(value) * float64(value)
		}
		want[i] = math.Sqrt(sum)
	}
	got := make([]float64, len(values))
	errs := make([]error, len(values))
	var wait sync.WaitGroup
	for i := range parameters {
		wait.Add(1)
		go func(i int) {
			defer wait.Done()
			context, err := NewCUDAContext()
			if err != nil {
				errs[i] = err
				return
			}
			defer context.Close()
			if err := parameters[i].Value.ensureGradGPU().CopyFromHost(floatBytes(values[i])); err != nil {
				errs[i] = err
				return
			}
			// maxNorm 0 leaves the gradients unscaled and stores the norm.
			ClipGradNorm([]Parameter{parameters[i]}, 0)
			got[i] = float64(GradientNormToHost())
		}(i)
	}
	wait.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", i, err)
		}
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-4 {
			t.Fatalf("worker %d norm %v, want %v", i, got[i], want[i])
		}
	}
}
