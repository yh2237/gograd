package autograd

import (
	"math"
	"sync"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestExecutionOptionsScopes(t *testing.T) {
	oldBF, oldAttention := BF16Autocast, AttentionAlgorithm
	defer func() { BF16Autocast, AttentionAlgorithm = oldBF, oldAttention }()
	BF16Autocast, AttentionAlgorithm = true, "flash"
	var context ExecutionContext
	other := NewExecutionContext()
	defaults := ExecutionOptions{AttentionAlgorithm: "auto"}
	if context.Options() != defaults || other.Options() != defaults {
		t.Fatal("explicit contexts inherited compatibility policy")
	}
	x := executionTensor(t, &context, []float32{1}, []int{}, tensor.CPU, true)
	if executionOptions(x) != defaults {
		t.Fatal("bound tensor did not resolve explicit policy")
	}
	neutral := Must([]float32{1}, []int{}, false)
	defer neutral.Close()
	if got := executionOptions(neutral); !got.BF16Autocast || got.AttentionAlgorithm != "flash" {
		t.Fatal("legacy policy was not preserved")
	}
	if err := context.SetOptions(ExecutionOptions{AttentionAlgorithm: "materialized"}); err != nil {
		t.Fatal(err)
	}
	context.Autocast(true, func() {
		if !context.Options().BF16Autocast || other.Options().BF16Autocast {
			t.Fatal("autocast scope not independent")
		}
		context.Autocast(false, func() {
			if context.Options().BF16Autocast {
				t.Fatal("nested disable failed")
			}
		})
		if !context.Options().BF16Autocast {
			t.Fatal("nested scope did not restore")
		}
		panicked := false
		func() {
			defer func() { panicked = recover() != nil }()
			context.Autocast(false, func() { panic("restore") })
		}()
		if !panicked || !context.Options().BF16Autocast {
			t.Fatal("panic did not restore precision")
		}
		// Precision scope restoration leaves unrelated policy changes intact.
		o := context.Options()
		o.AttentionAlgorithm = "flash"
		if err := context.SetOptions(o); err != nil {
			t.Fatal(err)
		}
	})
	if got := context.Options(); got.BF16Autocast || got.AttentionAlgorithm != "flash" {
		t.Fatal("scope restored unrelated settings incorrectly")
	}
	previous := context.Options()
	copy := context.Options()
	copy.BF16Autocast = !copy.BF16Autocast
	if context.Options() != previous {
		t.Fatal("Options exposed mutable context state")
	}
	if err := context.SetOptions(ExecutionOptions{BF16Autocast: true, AttentionAlgorithm: "invalid"}); err == nil {
		t.Fatal("invalid policy accepted")
	}
	if context.Options() != previous {
		t.Fatal("invalid update partially changed policy")
	}
	if err := context.SetOptions(ExecutionOptions{}); err != nil || context.Options() != defaults {
		t.Fatal("zero policy did not normalize")
	}
	scalar := Scalar(x, 2)
	defer scalar.Close()
	if scalar.ExecutionContext() != &context {
		t.Fatal("scalar device hint lost context")
	}
}

func TestExecutionOptionsCPUConcurrency(t *testing.T) {
	const workers = 8
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			context := NewExecutionContext()
			options := ExecutionOptions{BF16Autocast: worker%2 == 0, AttentionAlgorithm: "materialized"}
			if worker%3 == 0 {
				options.AttentionAlgorithm = "flash"
			}
			if err := context.SetOptions(options); err != nil {
				t.Error(err)
				return
			}
			x, err := context.New([]float32{1, 2, 3, 4}, []int{2, 2}, tensor.CPU, true)
			if err != nil {
				t.Error(err)
				return
			}
			defer x.Close()
			for step := 0; step < 30; step++ {
				if context.Options() != options || executionOptions(x) != options {
					t.Error("another worker changed policy")
					return
				}
				x.ZeroGrad()
				context.Autocast(!options.BF16Autocast, func() {
					loss := Sum(Mul(Transpose(x, 0, 1), Transpose(x, 0, 1)), 0, 1)
					if err := loss.Backward(); err != nil {
						t.Error(err)
					}
					loss.ReleaseGraph()
				})
				if context.Options() != options {
					t.Error("autocast did not restore worker policy")
					return
				}
				for i, v := range x.Grad {
					if v != 2*x.Data[i] {
						t.Error("CPU precision setting changed FP32 gradient")
						return
					}
				}
			}
		}(worker)
	}
	wg.Wait()
}

type executionPolicyResult struct {
	output []float32
	grads  [][]float32
	casts  int
}

func policyValues(n int, phase float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i)*.17+phase) * .3)
	}
	return v
}

// Compare a normal run with an identical forward whose context/compatibility
// policy is flipped before backward. This exercises cached, broadcast, single
// and irregular GEMMs and retained Conv1d/attention saved state.
func TestExecutionOptionsBackwardSnapshotCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	oldBF, oldAttention := BF16Autocast, AttentionAlgorithm
	defer func() { BF16Autocast, AttentionAlgorithm = oldBF, oldAttention; EnableGPUProfile(false) }()
	for _, kind := range []string{"regular", "broadcast", "single", "irregular", "conv1d", "attention", "producers", "binary_nd"} {
		for _, bf16 := range []bool{false, true} {
			name := kind + "/fp32"
			if bf16 {
				name = kind + "/bf16"
			}
			t.Run(name, func(t *testing.T) {
				run := func(flip bool) executionPolicyResult {
					context := NewExecutionContext()
					if err := context.SetOptions(ExecutionOptions{BF16Autocast: bf16, AttentionAlgorithm: "materialized"}); err != nil {
						t.Fatal(err)
					}
					// Explicit scope must ignore opposite globals at forward entry.
					BF16Autocast, AttentionAlgorithm = !bf16, "flash"
					var params []*Tensor
					makeTensor := func(shape []int) *Tensor {
						v := executionTensor(t, context, policyValues(numel(shape), float64(len(params))*.7+.2), shape, tensor.CUDA, true)
						params = append(params, v)
						return v
					}
					EnableGPUProfile(true)
					var y *Tensor
					switch kind {
					case "regular":
						y = MatMul(makeTensor([]int{2, 3, 4}), makeTensor([]int{2, 4, 5}))
					case "broadcast":
						y = MatMul(makeTensor([]int{2, 3, 4}), makeTensor([]int{4, 5}))
					case "single":
						y = MatMul(makeTensor([]int{3, 4}), makeTensor([]int{4, 5}))
					case "irregular":
						y = MatMul(makeTensor([]int{2, 1, 3, 4}), makeTensor([]int{1, 2, 4, 5}))
					case "conv1d":
						y = Conv1dGEMM(makeTensor([]int{1, 5, 2}), makeTensor([]int{3, 2, 3}), makeTensor([]int{3}), 1)
					case "attention":
						q, k, v, mask := makeTensor([]int{2, 2, 3, 4}), makeTensor([]int{2, 2, 5, 4}), makeTensor([]int{2, 2, 5, 3}), makeTensor([]int{1, 1, 3, 5})
						// Rectangular Q/K lengths catch batch-stride regressions too.
						y = ScaledDotProductAttention(q, k, v, mask)
					case "producers":
						x, w, b, bias, residual, head := makeTensor([]int{2, 3, 4}), makeTensor([]int{4}), makeTensor([]int{4}), makeTensor([]int{4}), makeTensor([]int{2, 3, 4}), makeTensor([]int{3, 5})
						h := BiasResidual(BiasGELU(LayerNorm(x, w, b, 1e-5), bias), bias, residual)
						h = Add(h, h)
						y = MatMul(Transpose(h, 1, 2).Contiguous(), head)
					case "binary_nd":
						a, b, w := makeTensor([]int{2, 2, 2, 2}), makeTensor([]int{2}), makeTensor([]int{2, 2})
						y = MatMul(Add(a, b), w)
					}
					out := conv2dHost(t, y, false)
					axes := make([]int, len(y.Shape))
					for i := range axes {
						axes[i] = i
					}
					loss := Mean(Mul(y, y), axes...)
					defer loss.ReleaseGraph()
					if flip {
						if err := context.SetOptions(ExecutionOptions{BF16Autocast: !bf16, AttentionAlgorithm: "flash"}); err != nil {
							t.Fatal(err)
						}
					}
					if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
						t.Fatal(err)
					}
					first := make([][]float32, len(params))
					for i, p := range params {
						first[i] = conv2dHost(t, p, true)
					}
					// A second pass after another configuration change must use the
					// same saved policy and add exactly one more leaf contribution.
					if flip {
						if err := context.SetOptions(ExecutionOptions{BF16Autocast: bf16}); err != nil {
							t.Fatal(err)
						}
					}
					if err := loss.Backward(); err != nil {
						t.Fatal(err)
					}
					for i, p := range params {
						want := append([]float32(nil), first[i]...)
						for j := range want {
							want[j] *= 2
						}
						conv2dClose(t, "retained policy gradient", conv2dHost(t, p, true), want, 1e-5, 1e-5)
					}
					casts := GPUProfile()["f32_to_bf16"].Count
					EnableGPUProfile(false)
					return executionPolicyResult{out, first, casts}
				}
				baseline, changed := run(false), run(true)
				conv2dClose(t, "policy snapshot output", changed.output, baseline.output, 1e-6, 1e-6)
				for i := range baseline.grads {
					conv2dClose(t, "policy snapshot gradient", changed.grads[i], baseline.grads[i], 1e-5, 1e-5)
				}
				if bf16 && changed.casts == 0 {
					t.Fatal("BF16 context did not run conversion")
				}
				if !bf16 && changed.casts != 0 {
					t.Fatal("FP32 context consulted BF16 compatibility policy")
				}
			})
		}
	}
}

func TestExecutionOptionsAutocastExitCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	context := NewExecutionContext()
	a := executionTensor(t, context, policyValues(24, .2), []int{2, 3, 4}, tensor.CUDA, true)
	b := executionTensor(t, context, policyValues(20, .7), []int{4, 5}, tensor.CUDA, true)
	defer a.Close()
	defer b.Close()
	var loss *Tensor
	context.Autocast(true, func() { loss = Sum(MatMul(a, b), 0, 1, 2) })
	defer loss.ReleaseGraph()
	if context.Options().BF16Autocast {
		t.Fatal("autocast did not exit")
	}
	EnableGPUProfile(true)
	defer EnableGPUProfile(false)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	if GPUProfile()["f32_to_bf16"].Count == 0 {
		t.Fatal("backward forgot forward autocast after scope exit")
	}
}

func TestExecutionOptionsLegacyBackwardSnapshotCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	old := BF16Autocast
	defer func() { BF16Autocast = old; EnableGPUProfile(false) }()
	for _, enabled := range []bool{false, true} {
		BF16Autocast = enabled
		a := conv2dTestTensor(t, policyValues(24, .2), []int{2, 3, 4}, tensor.CUDA, true)
		b := conv2dTestTensor(t, policyValues(20, .7), []int{4, 5}, tensor.CUDA, true)
		loss := Sum(MatMul(a, b), 0, 1, 2)
		BF16Autocast = !enabled
		EnableGPUProfile(true)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		casts := GPUProfile()["f32_to_bf16"].Count
		if (casts != 0) != enabled {
			t.Fatal("legacy backward read current global instead of saved policy")
		}
		loss.ReleaseGraph()
		a.Close()
		b.Close()
	}
}

func TestExecutionOptionsAttentionIsolationCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	oldBF, oldAttention := BF16Autocast, AttentionAlgorithm
	defer func() { BF16Autocast, AttentionAlgorithm = oldBF, oldAttention; EnableGPUProfile(false) }()
	contexts := []*ExecutionContext{NewExecutionContext(), NewExecutionContext(), NewExecutionContext()}
	modes := []string{"materialized", "flash", "auto"}
	for i, c := range contexts {
		if err := c.SetOptions(ExecutionOptions{AttentionAlgorithm: modes[i]}); err != nil {
			t.Fatal(err)
		}
	}
	run := func(context *ExecutionContext, device tensor.Device) [][]float32 {
		shapes := [][]int{{2, 2, 3, 4}, {2, 2, 5, 4}, {2, 2, 5, 3}, {1, 1, 3, 5}}
		params := make([]*Tensor, len(shapes))
		for i, shape := range shapes {
			params[i] = executionTensor(t, context, policyValues(numel(shape), float64(i)*.7+.2), shape, device, true)
			defer params[i].Close()
		}
		y := ScaledDotProductAttention(params[0], params[1], params[2], params[3])
		out := [][]float32{conv2dHost(t, y, false)}
		loss := Sum(y, 0, 1, 2, 3)
		defer loss.ReleaseGraph()
		// Other contexts and compatibility settings must not redirect either
		// this forward or its already-selected backward implementation.
		if device == tensor.CUDA {
			if err := context.SetOptions(ExecutionOptions{BF16Autocast: true, AttentionAlgorithm: "flash"}); err != nil {
				t.Fatal(err)
			}
		}
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		for _, p := range params {
			out = append(out, conv2dHost(t, p, true))
		}
		return out
	}
	want := run(NewExecutionContext(), tensor.CPU)
	for i, c := range contexts {
		BF16Autocast = true
		AttentionAlgorithm = "flash"
		if modes[i] == "flash" {
			AttentionAlgorithm = "materialized"
		}
		EnableGPUProfile(true)
		got := run(c, tensor.CUDA)
		profile := GPUProfile()
		EnableGPUProfile(false)
		flash := profile["flash_attention_tiled_f"].Count > 0 && profile["flash_attention_tiled_b"].Count > 0
		if flash != (modes[i] == "flash") {
			t.Fatalf("%s selected wrong attention implementation", modes[i])
		}
		if profile["f32_to_bf16"].Count != 0 {
			t.Fatalf("%s backward forgot FP32 forward policy", modes[i])
		}
		for component := range want {
			conv2dClose(t, modes[i]+" attention CPU parity", got[component], want[component], 1e-5, 1e-4)
		}
	}
}

func TestExecutionOptionsGraphCaptureCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	old := BF16Autocast
	BF16Autocast = false
	defer func() { BF16Autocast = old }()
	context := NewExecutionContext()
	a := executionTensor(t, context, policyValues(24, .2), []int{2, 3, 4}, tensor.CUDA, true)
	defer a.Close()
	b := executionTensor(t, context, policyValues(20, .7), []int{4, 5}, tensor.CUDA, true)
	defer b.Close()
	step := func() error {
		a.ZeroGrad()
		b.ZeroGrad()
		var loss *Tensor
		context.Autocast(true, func() { loss = Sum(MatMul(a, b), 0, 1, 2) })
		defer loss.ReleaseGraph()
		return loss.Backward()
	}
	if err := step(); err != nil {
		t.Fatal(err)
	}
	wantA, wantB := conv2dHost(t, a, true), conv2dHost(t, b, true)
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	if err := SetBLASStream(stream); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := SetBLASStream(nil); err != nil {
			t.Error(err)
		}
	}()
	graph, err := cuda.Capture(stream, step)
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if context.Options().BF16Autocast {
		t.Fatal("captured autocast did not exit")
	}
	if err := context.SetOptions(ExecutionOptions{AttentionAlgorithm: "flash"}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err := graph.Launch(); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "captured context dA", conv2dHost(t, a, true), wantA, 1e-5, 1e-5)
	conv2dClose(t, "captured context dB", conv2dHost(t, b, true), wantB, 1e-5, 1e-5)
}
