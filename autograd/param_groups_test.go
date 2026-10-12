package autograd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type paramGroupFixture struct {
	Name   string `json:"name"`
	Kind   string `json:"kind"`
	Params []struct {
		Name  string    `json:"name"`
		Value []float32 `json:"value"`
	} `json:"params"`
	Groups []struct {
		Name        string   `json:"name"`
		Params      []string `json:"params"`
		LR          float32  `json:"lr"`
		WeightDecay float32  `json:"weight_decay"`
	} `json:"groups"`
	Steps []struct {
		Gradients [][]float32   `json:"gradients"`
		Values    [][]float32   `json:"values"`
		Buffers   [][]float32   `json:"buffers"`
		Moments   [][][]float32 `json:"moments"`
	} `json:"steps"`
}

func paramGroupFixtures(t *testing.T, kind string) []paramGroupFixture {
	t.Helper()
	raw, err := os.ReadFile("../testdata/param_groups.json")
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Cases []paramGroupFixture `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	out := make([]paramGroupFixture, 0, len(file.Cases))
	for _, c := range file.Cases {
		if c.Kind == kind {
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		t.Fatalf("no %s parameter group fixtures", kind)
	}
	return out
}

// groupFixtureSpecs converts recorded absolute group learning rates into scales
// of the first group's rate, so one scheduler step still drives every group.
func groupFixtureSpecs(f paramGroupFixture, params []Parameter) []ParameterGroupSpec {
	specs := make([]ParameterGroupSpec, 0, len(f.Groups))
	for _, group := range f.Groups {
		spec := ParameterGroupSpec{Name: group.Name, WeightDecay: group.WeightDecay, LRScale: group.LR / f.Groups[0].LR}
		for _, p := range params {
			for _, name := range group.Params {
				if p.Name == name {
					spec.Params = append(spec.Params, p)
				}
			}
		}
		specs = append(specs, spec)
	}
	return specs
}

func groupFixtureParams(t *testing.T, f paramGroupFixture, device tensor.Device) []Parameter {
	t.Helper()
	params := make([]Parameter, len(f.Params))
	for i, p := range f.Params {
		params[i] = Parameter{Name: p.Name, Value: conv2dTestTensor(t, p.Value, []int{len(p.Value)}, device, true)}
	}
	return params
}

func TestParameterGroupSGDPyTorchParity(t *testing.T) {
	for _, f := range paramGroupFixtures(t, "sgd") {
		t.Run(f.Name, func(t *testing.T) {
			pool2dDevices(t, func(t *testing.T, device tensor.Device) {
				pool2dContext(t, device)
				params := groupFixtureParams(t, f, device)
				optimizer, err := NewSGDGroups(params, SGDOptions{LR: f.Groups[0].LR, Momentum: .9, Nesterov: true}, groupFixtureSpecs(f, params))
				if err != nil {
					t.Fatal(err)
				}
				defer optimizer.Close()
				EnableGPUProfile(device == tensor.CUDA)
				defer EnableGPUProfile(false)
				for step, entry := range f.Steps {
					optimizer.ZeroGrad()
					for i, grad := range entry.Gradients {
						setSGDGradient(t, params[i].Value, grad)
					}
					optimizer.Step()
					for i, p := range params {
						conv2dClose(t, "value", conv2dHost(t, p.Value, false), entry.Values[i], 5e-7, 3e-6)
					}
					state, err := optimizer.State()
					if err != nil {
						t.Fatal(err)
					}
					if state.StepCount != step+1 {
						t.Fatalf("step count %d != %d", state.StepCount, step+1)
					}
					if len(state.Groups) != len(f.Groups) {
						t.Fatalf("state groups %d != %d", len(state.Groups), len(f.Groups))
					}
					for i, group := range f.Groups {
						if state.Groups[i].Name != group.Name || !reflect.DeepEqual(state.Groups[i].Names, group.Params) {
							t.Fatalf("group %d state %+v", i, state.Groups[i])
						}
						if state.Groups[i].LRScale != group.LR/f.Groups[0].LR || state.Groups[i].WeightDecay != group.WeightDecay {
							t.Fatalf("group %d scale/decay %+v", i, state.Groups[i])
						}
					}
					for i, buffer := range entry.Buffers {
						if buffer == nil {
							if state.Initialized[i] {
								t.Fatalf("parameter %d momentum initialized", i)
							}
							continue
						}
						conv2dClose(t, "momentum", state.Buffers[i], buffer, 5e-7, 3e-6)
					}
				}
				if device == tensor.CUDA && GPUProfile()["sgd_update"].Count == 0 {
					t.Fatal("grouped SGD left CUDA")
				}
			})
		})
	}
}

func TestParameterGroupAdamWPyTorchParity(t *testing.T) {
	for _, f := range paramGroupFixtures(t, "adamw") {
		t.Run(f.Name, func(t *testing.T) {
			pool2dDevices(t, func(t *testing.T, device tensor.Device) {
				pool2dContext(t, device)
				params := groupFixtureParams(t, f, device)
				optimizer, err := NewAdamWGroups(params, f.Groups[0].LR, groupFixtureSpecs(f, params))
				if err != nil {
					t.Fatal(err)
				}
				defer optimizer.Close()
				EnableGPUProfile(device == tensor.CUDA)
				defer EnableGPUProfile(false)
				for step, entry := range f.Steps {
					optimizer.ZeroGrad()
					for i, grad := range entry.Gradients {
						setSGDGradient(t, params[i].Value, grad)
					}
					optimizer.Step()
					for i, p := range params {
						conv2dClose(t, "value", conv2dHost(t, p.Value, false), entry.Values[i], 5e-7, 3e-6)
					}
					state, err := optimizer.State()
					if err != nil {
						t.Fatal(err)
					}
					if state.StepCount != step+1 {
						t.Fatalf("step count %d != %d", state.StepCount, step+1)
					}
					for i, group := range f.Groups {
						if state.Groups[i].Name != group.Name || state.Groups[i].LRScale != group.LR/f.Groups[0].LR || state.Groups[i].WeightDecay != group.WeightDecay {
							t.Fatalf("group %d %+v", i, state.Groups[i])
						}
						if !reflect.DeepEqual(state.Groups[i].Names, group.Params) {
							t.Fatalf("group %d names %v", i, state.Groups[i].Names)
						}
					}
					for i, moment := range entry.Moments {
						if moment == nil {
							t.Fatalf("parameter %d has no moment", i)
						}
						conv2dClose(t, "m", state.Moments[i].M, moment[0], 5e-7, 3e-6)
						conv2dClose(t, "v", state.Moments[i].V, moment[1], 5e-8, 3e-7)
					}
				}
				// Device parity comes from the shared value comparison; the CUDA
				// optimizer keeps its moments on the device rather than in host
				// slices, which this check observes.
				if device == tensor.CUDA && (len(optimizer.mGPU) != len(params) || len(optimizer.M) != 0) {
					t.Fatal("grouped AdamW did not keep moments on CUDA")
				}
			})
		})
	}
}

func TestParameterGroupCUDACapture(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	pool2dContext(t, tensor.CUDA)
	initial := []float32{.3, -.7, .2, .4}
	captured := Parameter{"weight", conv2dTestTensor(t, initial, []int{2, 2}, tensor.CUDA, true)}
	other := Parameter{"weight", conv2dTestTensor(t, initial, []int{2, 2}, tensor.CUDA, true)}
	shift := Parameter{"shift", conv2dTestTensor(t, initial, []int{2, 2}, tensor.CUDA, true)}
	shiftCopy := Parameter{"shift", conv2dTestTensor(t, initial, []int{2, 2}, tensor.CUDA, true)}
	options := SGDOptions{LR: .03, Momentum: .9}
	groupSpecs := func(first, second Parameter) []ParameterGroupSpec {
		return []ParameterGroupSpec{
			{Name: "slow", Params: []Parameter{first}, LRScale: .5, WeightDecay: .02},
			{Name: "fast", Params: []Parameter{second}},
		}
	}
	graphOptions, err := NewSGDGroups([]Parameter{captured, shift}, options, groupSpecs(captured, shift))
	if err != nil {
		t.Fatal(err)
	}
	defer graphOptions.Close()
	eagerOptions, err := NewSGDGroups([]Parameter{other, shiftCopy}, options, groupSpecs(other, shiftCopy))
	if err != nil {
		t.Fatal(err)
	}
	defer eagerOptions.Close()
	// Warm the CUDA kernels once outside capture.
	warm := Parameter{"warm", conv2dTestTensor(t, initial, []int{2, 2}, tensor.CUDA, true)}
	precompile, err := NewSGD([]Parameter{warm}, SGDOptions{LR: .03, Momentum: .9})
	if err != nil {
		t.Fatal(err)
	}
	setSGDGradient(t, warm.Value, []float32{.2, -.3, .1, .4})
	precompile.Step()
	precompile.Close()
	setSGDGradient(t, captured.Value, []float32{.2, -.3, .1, .4})
	setSGDGradient(t, shift.Value, []float32{.2, -.3, .1, .4})
	if err := graphOptions.PrepareGraph(); err != nil {
		t.Fatal(err)
	}
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	graph, err := cuda.Capture(stream, func() error { graphOptions.Step(); return nil })
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
	for range 3 {
		eagerOptions.ZeroGrad()
		setSGDGradient(t, other.Value, []float32{.2, -.3, .1, .4})
		setSGDGradient(t, shiftCopy.Value, []float32{.2, -.3, .1, .4})
		eagerOptions.Step()
	}
	replayed, err := graphOptions.State()
	if err != nil {
		t.Fatal(err)
	}
	live, err := eagerOptions.State()
	if err != nil {
		t.Fatal(err)
	}
	if replayed.StepCount != 3 {
		t.Fatalf("captured step count %d", replayed.StepCount)
	}
	conv2dClose(t, "captured slow group", conv2dHost(t, captured.Value, false), conv2dHost(t, other.Value, false), 1e-6, 1e-6)
	conv2dClose(t, "captured fast group", conv2dHost(t, shift.Value, false), conv2dHost(t, shiftCopy.Value, false), 1e-6, 1e-6)
	for i := range replayed.Buffers {
		conv2dClose(t, "captured momentum", replayed.Buffers[i], live.Buffers[i], 1e-6, 1e-6)
	}
	// The two groups must stay apart, so replay cannot be explained by a single
	// baked base learning rate.
	if conv2dHost(t, captured.Value, false)[0] == conv2dHost(t, shift.Value, false)[0] {
		t.Fatal("captured replay treated both groups alike")
	}
}

func TestParameterGroupTrainingCheckpoint(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		newRun := func() (*Module, CheckpointOptimizer, *OneCycle) {
			weight := conv2dTestTensor(t, []float32{.3, -.7}, []int{2}, device, true)
			bias := conv2dTestTensor(t, []float32{.1, .2}, []int{2}, device, true)
			module := &Module{Parameters: []Parameter{{Name: "weight", Value: weight}, {Name: "bias", Value: bias}}}
			specs := []ParameterGroupSpec{
				{Name: "decay", Params: []Parameter{module.NamedParameters()[0]}, LRScale: .5, WeightDecay: .02},
				{Name: "plain", Params: []Parameter{module.NamedParameters()[1]}},
			}
			optimizer, err := NewSGDGroups(module.NamedParameters(), SGDOptions{LR: .03, Momentum: .9}, specs)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(optimizer.Close)
			return module, optimizer, NewOneCycle(.03, 8, .3)
		}
		module, optimizer, schedule := newRun()
		optimizer.SetLearningRate(float32(schedule.LR()))
		step := func(module *Module, optimizer CheckpointOptimizer, schedule *OneCycle) {
			optimizer.ZeroGrad()
			loss := Sum(Mul(module.Parameters[0].Value, module.Parameters[0].Value), 0)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			loss.ReleaseGraph()
			optimizer.Step()
			schedule.Step(optimizer)
		}
		for range 3 {
			step(module, optimizer, schedule)
		}
		path := filepath.Join(t.TempDir(), "grouped.safetensors")
		if err := SaveTrainingCheckpoint(path, module, optimizer, schedule, map[string]string{"run": "grouped"}); err != nil {
			t.Fatal(err)
		}
		// A differently grouped optimizer must be rejected before touching it.
		rejected, _, wrongSchedule := newRun()
		flat := NewAdamW(rejected.NamedParameters(), .03, 0)
		defer flat.Close()
		before := rejected.StateDict()
		if _, err := LoadTrainingCheckpoint(path, rejected, flat, wrongSchedule); err == nil {
			t.Fatal("group mismatch accepted")
		}
		if !reflect.DeepEqual(before, rejected.StateDict()) {
			t.Fatal("rejected checkpoint mutated the model")
		}
		restored, restoredOptimizer, restoredSchedule := newRun()
		meta, err := LoadTrainingCheckpoint(path, restored, restoredOptimizer, restoredSchedule)
		if err != nil || meta["run"] != "grouped" {
			t.Fatalf("grouped checkpoint restore: %v", err)
		}
		for i := 3; i < 8; i++ {
			step(module, optimizer, schedule)
			step(restored, restoredOptimizer, restoredSchedule)
		}
		first, err := optimizer.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		second, err := restoredOptimizer.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(module.StateDict(), restored.StateDict()) || !reflect.DeepEqual(first.SGD, second.SGD) {
			t.Fatalf("grouped resume diverged:\n%+v\n%+v", first.SGD, second.SGD)
		}
		if !reflect.DeepEqual(schedule, restoredSchedule) {
			t.Fatal("grouped resume diverged in the schedule")
		}
	})
}

// cloneGroupState deep-copies group state so a test can corrupt one field
// without touching the optimizer's own snapshot.
func cloneGroupState(s SGDState) SGDState {
	out := s
	out.Groups = make([]OptimizerGroupState, len(s.Groups))
	for i, group := range s.Groups {
		clone := group
		clone.Names = append([]string(nil), group.Names...)
		out.Groups[i] = clone
	}
	return out
}

func TestParameterGroupCheckpointRoundTrip(t *testing.T) {
	base := testParameter(t, []float32{.5, -.25, 1.5})
	other := testParameter(t, []float32{.3, -.7})
	base.Name, other.Name = "weight", "bias"
	params := []Parameter{base, other}
	options := SGDOptions{LR: .05, Momentum: .9}
	specs := []ParameterGroupSpec{
		{Name: "decay", Params: []Parameter{base}, LRScale: .5, WeightDecay: .02},
		{Name: "plain", Params: []Parameter{other}},
	}
	optimizer, err := NewSGDGroups(params, options, specs)
	if err != nil {
		t.Fatal(err)
	}
	defer optimizer.Close()
	for range 3 {
		optimizer.ZeroGrad()
		base.Value.Grad = []float32{.2, -.1, .3}
		other.Value.Grad = []float32{-.2, .1}
		optimizer.Step()
	}
	want, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "sgd.safetensors")
	if err := optimizer.SaveSafeTensors(path); err != nil {
		t.Fatal(err)
	}
	// A plain optimizer with the wrong group layout must be rejected.
	flat, err := NewSGD(params, SGDOptions{LR: .05, Momentum: .9})
	if err != nil {
		t.Fatal(err)
	}
	defer flat.Close()
	if err := flat.LoadSafeTensors(path); err == nil {
		t.Fatal("ungrouped optimizer accepted grouped state")
	}
	// Same layout restores groups, scales, buffers and step count.
	restored, err := NewSGDGroups(params, options, specs)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if err := restored.LoadSafeTensors(path); err != nil {
		t.Fatal(err)
	}
	got, err := restored.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("restored state differs:\n%+v\n%+v", got, want)
	}
	// Continuing after restore keeps the second group at half the base rate.
	restored.SetLearningRate(.1)
	snapshot, err := restored.State()
	if err != nil {
		t.Fatal(err)
	}
	beforeWeight := conv2dHost(t, base.Value, false)
	beforeBias := conv2dHost(t, other.Value, false)
	restored.ZeroGrad()
	base.Value.Grad = []float32{.1, .1, .1}
	other.Value.Grad = []float32{.1, .1}
	restored.Step()
	afterWeight := conv2dHost(t, base.Value, false)
	afterBias := conv2dHost(t, other.Value, false)
	// Group "decay" runs at half the base LR with decay .02; group "plain" runs
	// at the base LR without decay. Momentum .9 is already initialized, so each
	// buffer becomes 0.9*buffer + 0.1*(grad + decay*param).
	for i := range afterBias {
		// Group "decay" runs at half the restored base LR (.05) with decay .02;
		// group "plain" runs at the base LR (.1) without decay. Momentum .9 with
		// zero dampening adds the full decay-adjusted gradient each step.
		weightBuffer := .9*float64(snapshot.Buffers[0][i]) + (.1 + .02*float64(beforeWeight[i]))
		biasBuffer := .9*float64(snapshot.Buffers[1][i]) + .1
		wantWeight := beforeWeight[i] - float32(.05*weightBuffer)
		wantBias := beforeBias[i] - float32(.1*biasBuffer)
		conv2dClose(t, "scaled weight", []float32{afterWeight[i]}, []float32{wantWeight}, 5e-7, 3e-6)
		conv2dClose(t, "unscaled bias", []float32{afterBias[i]}, []float32{wantBias}, 5e-7, 3e-6)
	}
}

func TestParameterGroupScaleAndScheduler(t *testing.T) {
	start := []float32{1, 2}
	base := testParameter(t, start)
	other := testParameter(t, start)
	base.Name, other.Name = "weight", "shift"
	params := []Parameter{base, other}
	specs := []ParameterGroupSpec{
		{Name: "slow", Params: []Parameter{base}, LRScale: .5},
		{Name: "fast", Params: []Parameter{other}},
	}
	optimizer, err := NewSGDGroups(params, SGDOptions{LR: .2}, specs)
	if err != nil {
		t.Fatal(err)
	}
	defer optimizer.Close()
	optimizer.ZeroGrad()
	base.Value.Grad = []float32{.4, -.4}
	other.Value.Grad = []float32{.4, -.4}
	optimizer.Step()
	// Without momentum the update is exactly LR*(grad); the scaled group runs at
	// half the base rate and moves half as far in both directions.
	conv2dClose(t, "slow group", conv2dHost(t, base.Value, false), []float32{start[0] - .2*.5*.4, start[1] + .2*.5*.4}, 0, 0)
	conv2dClose(t, "fast group", conv2dHost(t, other.Value, false), []float32{start[0] - .2*.4, start[1] + .2*.4}, 0, 0)
	// A scheduler only sees the base rate; both groups keep their ratio.
	if optimizer.LearningRate() != .2 {
		t.Fatalf("learning rate %v", optimizer.LearningRate())
	}
	optimizer.SetLearningRate(.4)
	if optimizer.LearningRate() != .4 {
		t.Fatalf("learning rate %v", optimizer.LearningRate())
	}
	state, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	if state.Options.LR != .4 {
		t.Fatalf("base LR not recorded: %+v", state.Options)
	}
	optimizer.ZeroGrad()
	base.Value.Grad = []float32{1, 1}
	other.Value.Grad = []float32{1, 1}
	optimizer.Step()
	slow := conv2dHost(t, base.Value, false)
	fast := conv2dHost(t, other.Value, false)
	// The scaled group keeps moving at half the base rate after its change.
	conv2dClose(t, "slow group", slow, []float32{start[0] - .2*.5*.4 - .4*.5, start[1] + .2*.5*.4 - .4*.5}, 0, 0)
	conv2dClose(t, "fast group", fast, []float32{start[0] - .2*.4 - .4, start[1] + .2*.4 - .4}, 0, 0)
}

func TestParameterGroupStateValidation(t *testing.T) {
	weight := testParameter(t, []float32{1, 2})
	bias := testParameter(t, []float32{3, 4})
	weight.Name, bias.Name = "weight", "bias"
	defer weight.Value.Close()
	defer bias.Value.Close()
	params := []Parameter{weight, bias}
	specs := []ParameterGroupSpec{
		{Name: "decay", Params: []Parameter{weight}, LRScale: .5, WeightDecay: .02},
		{Name: "plain", Params: []Parameter{bias}},
	}
	optimizer, err := NewSGDGroups(params, SGDOptions{LR: .1, Momentum: .9}, specs)
	if err != nil {
		t.Fatal(err)
	}
	defer optimizer.Close()
	if _, err := NewSGDGroups(params, SGDOptions{LR: .1}, []ParameterGroupSpec{{Name: "one", Params: []Parameter{weight}}}); err == nil {
		t.Fatal("incomplete partition accepted")
	}
	if _, err := NewSGDGroups(params, SGDOptions{LR: .1}, []ParameterGroupSpec{{Name: "one", Params: params}, {Name: "one", Params: params}}); err == nil {
		t.Fatal("duplicate group name accepted")
	}
	if _, err := NewSGDGroups(params, SGDOptions{LR: .1}, []ParameterGroupSpec{{Name: "one", Params: []Parameter{weight, weight}}}); err == nil {
		t.Fatal("duplicate parameter accepted")
	}
	if _, err := NewSGDGroups(params, SGDOptions{LR: .1}, []ParameterGroupSpec{{Name: "one", Params: nil}}); err == nil {
		t.Fatal("empty group accepted")
	}
	if _, err := NewSGDGroups(params, SGDOptions{LR: .1}, []ParameterGroupSpec{{Name: "one", Params: params, LRScale: -1}}); err == nil {
		t.Fatal("negative scale accepted")
	}
	if _, err := NewSGDGroups(params, SGDOptions{LR: .1}, []ParameterGroupSpec{{Name: "one", Params: params, WeightDecay: -1}}); err == nil {
		t.Fatal("negative decay accepted")
	}
	// Renaming a parameter or a group must be rejected before any state moves.
	state, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	broken := cloneGroupState(state)
	broken.Groups[0].Names = []string{"gain"}
	broken.Groups[1].Names = []string{"weight"}
	if err := optimizer.LoadState(broken); err == nil {
		t.Fatal("renamed group membership accepted")
	}
	broken = cloneGroupState(state)
	broken.Groups[1].Name = "other"
	if err := optimizer.LoadState(broken); err == nil {
		t.Fatal("renamed group accepted")
	}
	broken = cloneGroupState(state)
	broken.Groups = broken.Groups[:1]
	if err := optimizer.LoadState(broken); err == nil {
		t.Fatal("missing group accepted")
	}
	broken = cloneGroupState(state)
	broken.Groups[0].LRScale = -1
	if err := optimizer.LoadState(broken); err == nil {
		t.Fatal("negative group scale accepted")
	}
	broken = cloneGroupState(state)
	broken.Groups[1].WeightDecay = -1
	if err := optimizer.LoadState(broken); err == nil {
		t.Fatal("negative group decay accepted")
	}
	after, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(after, state) {
		t.Fatalf("rejected state mutated the optimizer:\n%+v\n%+v", after, state)
	}
	// Legacy state without groups only fits a single-group optimizer.
	legacy := state
	legacy.Groups = nil
	legacy.Options.WeightDecay = .07
	flat, err := NewSGD(params, SGDOptions{LR: .1, Momentum: .9, WeightDecay: .3})
	if err != nil {
		t.Fatal(err)
	}
	defer flat.Close()
	if err := flat.LoadState(legacy); err != nil {
		t.Fatal(err)
	}
	flatState, err := flat.State()
	if err != nil {
		t.Fatal(err)
	}
	if flatState.Options.WeightDecay != .07 || flatState.Groups[0].WeightDecay != .07 {
		t.Fatalf("legacy decay not applied: %+v", flatState.Groups[0])
	}
	if err := optimizer.LoadState(legacy); err == nil {
		t.Fatal("legacy state loaded into grouped optimizer")
	}
	stats, err := PartitionParameters(params, func(name string) string {
		if name == "bias" {
			return "plain"
		}
		return "decay"
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(stats) != 2 || stats[0].Name != "decay" || len(stats[0].Params) != 1 || stats[1].Name != "plain" || len(stats[1].Params) != 1 {
		t.Fatalf("partitioned specs %+v", stats)
	}
	if _, err := PartitionParameters(params, func(string) string { return "" }); err == nil {
		t.Fatal("unassigned partition accepted")
	}
	if _, err := PartitionParameters(nil, func(string) string { return "x" }); err == nil {
		t.Fatal("empty partition accepted")
	}
}
