package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func auditCUDAContext(t *testing.T) *CUDAContext {
	t.Helper()
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	return ctx
}

func TestAuditGroupNormLargeOffsetCPU(t *testing.T) {
	auditGroupNormLargeOffset(t, tensor.CPU)
}

func TestAuditGroupNormLargeOffsetCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	auditGroupNormLargeOffset(t, tensor.CUDA)
}

func auditGroupNormLargeOffset(t *testing.T, device tensor.Device) {
	x := conv2dTestTensor(t, []float32{9999, 10001}, []int{1, 1, 2}, device, true)
	w := conv2dTestTensor(t, []float32{1.5, -.5}, []int{2}, device, true)
	b := conv2dTestTensor(t, []float32{.25, -.75}, []int{2}, device, true)
	up := conv2dTestTensor(t, []float32{2, -3}, []int{1, 1, 2}, device, false)
	y := GroupNorm(x, w, b, 1, .25)
	inv := float32(1 / math.Sqrt(1.25))
	conv2dClose(t, "groupnorm forward", conv2dHost(t, y, false), []float32{.25 - 1.5*inv, -.75 - .5*inv}, 3e-5, 3e-5)
	loss := Sum(Mul(y, up), 0, 1, 2)
	defer loss.ReleaseGraph()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "groupnorm dx", conv2dHost(t, x, true), []float32{.15 * inv, -.15 * inv}, 3e-5, 3e-5)
	conv2dClose(t, "groupnorm dw", conv2dHost(t, w, true), []float32{-2 * inv, -3 * inv}, 3e-5, 3e-5)
	conv2dClose(t, "groupnorm db", conv2dHost(t, b, true), []float32{2, -3}, 3e-5, 3e-5)
}

func TestAuditCrossEntropyLargeOffsetCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	// At this magnitude mx+log(2) rounds to mx in float32. Subtracting
	// the target first must retain the log-normalizer and its gradient.
	x := conv2dTestTensor(t, []float32{1e8, 1e8, -1e8, -1e8}, []int{2, 2}, tensor.CUDA, true)
	loss := CrossEntropy(x, []int{0, 1}, -1)
	defer loss.ReleaseGraph()
	conv2dClose(t, "CE loss", conv2dHost(t, loss, false), []float32{float32(math.Log(2))}, 2e-6, 2e-6)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "CE dx", conv2dHost(t, x, true), []float32{-.25, .25, .25, -.25}, 2e-6, 2e-6)
}

func TestAuditAdamWCapturedDecayCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	makeOptimizer := func() *AdamW {
		params := []Parameter{
			{Name: "fast", Value: conv2dTestTensor(t, []float32{2, -3}, []int{2}, tensor.CUDA, true)},
			{Name: "slow", Value: conv2dTestTensor(t, []float32{2, -3}, []int{2}, tensor.CUDA, true)},
		}
		for _, p := range params {
			p.Value.ensureGradGPU() // Explicitly present, zero gradients.
		}
		o, err := NewAdamWGroups(params, .2, []ParameterGroupSpec{
			{Name: "fast", Params: params[:1], WeightDecay: 2, LRScale: 1},
			{Name: "slow", Params: params[1:], WeightDecay: 2, LRScale: .5},
		})
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	captured, eager := makeOptimizer(), makeOptimizer()
	defer captured.Close()
	defer eager.Close()
	captured.Step()
	eager.Step()
	if err := captured.PrepareGraph(); err != nil {
		t.Fatal(err)
	}
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	graph, err := cuda.Capture(stream, func() error { captured.Step(); return nil })
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	for range 3 {
		if err := graph.Launch(); err != nil {
			t.Fatal(err)
		}
		eager.Step()
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	for i, p := range captured.Params {
		got := conv2dHost(t, p.Value, false)
		conv2dClose(t, "captured/eager decay", got, conv2dHost(t, eager.Params[i].Value, false), 2e-6, 2e-6)
		factor := float32(math.Pow([]float64{.6, .8}[i], 4))
		conv2dClose(t, "analytic decay", got, []float32{2 * factor, -3 * factor}, 2e-6, 2e-6)
	}
}

func TestAuditClipAbsentGradientsCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	active := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CUDA, true)
	unused := conv2dTestTensor(t, []float32{7, 8}, []int{2}, tensor.CUDA, true)
	frozen := conv2dTestTensor(t, []float32{9, 10}, []int{2}, tensor.CUDA, false)
	params := []Parameter{{Name: "active", Value: active}, {Name: "unused", Value: unused}, {Name: "frozen", Value: frozen}}
	if err := active.ensureGradGPU().CopyFromHost(floatBytes([]float32{3, 4})); err != nil {
		t.Fatal(err)
	}
	opt, err := NewSGD(params, SGDOptions{LR: .1, Momentum: .9, WeightDecay: .5})
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Close()
	ClipGradNorm(params, 1)
	if unused.gradBuf != nil || frozen.gradBuf != nil {
		t.Fatal("clipping created an absent gradient")
	}
	if got := GradientNormToHost(); math.Abs(float64(got-5)) > 1e-5 {
		t.Fatalf("gradient norm %g, want 5", got)
	}
	plan := getClipPlan(params)
	if len(plan.ptrs) != 1 || plan.ptrs[0] != active.gradBuf.Pointer() {
		t.Fatalf("clip plan did not filter absent gradients: %v", plan.ptrs)
	}
	norm := clipStateForThread().norm
	ClipGradNorm(params, 0)
	if getClipPlan(params) != plan || clipStateForThread().norm != norm {
		t.Fatal("unchanged gradients invalidated the warmed clip plan/norm")
	}
	opt.Step()
	state, err := opt.State()
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{1, 2} {
		if state.Initialized[i] || len(state.Buffers[i]) != 0 {
			t.Fatalf("clipping initialized unused SGD momentum %d", i)
		}
	}
	conv2dClose(t, "unused weight", conv2dHost(t, unused, false), []float32{7, 8}, 0, 0)
	conv2dClose(t, "frozen weight", conv2dHost(t, frozen, false), []float32{9, 10}, 0, 0)
	unused.ensureGradGPU()
	if next := getClipPlan(params); next == plan || len(next.ptrs) != 2 {
		t.Fatal("clip cache did not observe newly present gradient")
	}
	// Removing a gradient must invalidate the cache too.
	unused.gradBuf.Free()
	unused.gradBuf = nil
	if next := getClipPlan(params); len(next.ptrs) != 1 {
		t.Fatal("clip cache retained removed gradient")
	}
	ClipGradNorm(params[1:], 1)
	if got := GradientNormToHost(); got != 0 {
		t.Fatalf("all-absent gradient norm %g, want 0", got)
	}
}

func TestAuditClipCaptureUnusedFrozenCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	active := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CUDA, true)
	unused := conv2dTestTensor(t, []float32{7, 8}, []int{2}, tensor.CUDA, true)
	frozen := conv2dTestTensor(t, []float32{9, 10}, []int{2}, tensor.CUDA, false)
	params := []Parameter{{Name: "active", Value: active}, {Name: "unused", Value: unused}, {Name: "frozen", Value: frozen}}
	opt, err := NewSGD(params, SGDOptions{LR: .1, Momentum: .9, WeightDecay: .5})
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Close()
	step := func() error {
		opt.ZeroGrad()
		loss := Sum(active, 0)
		if err := loss.Backward(); err != nil {
			return err
		}
		ClipGradNorm(params, 1)
		opt.Step()
		loss.ReleaseGraph()
		return nil
	}
	if err := step(); err != nil {
		t.Fatal(err)
	}
	plan, norm := getClipPlan(params), clipStateForThread().norm
	if err := opt.PrepareGraph(); err != nil {
		t.Fatal(err)
	}
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	graph, err := cuda.Capture(stream, step)
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	for range 3 {
		if err := graph.Launch(); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if getClipPlan(params) != plan || clipStateForThread().norm != norm {
		t.Fatal("capture changed stable gradient addresses or norm storage")
	}
	if unused.gradBuf != nil || frozen.gradBuf != nil {
		t.Fatal("capture allocated absent gradients")
	}
	state, err := opt.State()
	if err != nil {
		t.Fatal(err)
	}
	if state.Initialized[1] || state.Initialized[2] {
		t.Fatal("captured clipping initialized unused/frozen SGD state")
	}
	conv2dClose(t, "captured unused", conv2dHost(t, unused, false), []float32{7, 8}, 0, 0)
	conv2dClose(t, "captured frozen", conv2dHost(t, frozen, false), []float32{9, 10}, 0, 0)
}
