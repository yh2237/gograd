package autograd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type sgdFixture struct {
	Name    string
	Options SGDOptions
	Initial [][]float32
	Steps   []struct {
		LR                         float32
		Gradients, Values, Buffers [][]float32
	}
}

func sgdFixtures(t *testing.T) []sgdFixture {
	t.Helper()
	bytes, err := os.ReadFile("../testdata/sgd.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct{ Cases []sgdFixture }
	if err := json.Unmarshal(bytes, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("empty SGD fixtures")
	}
	return f.Cases
}
func setSGDGradient(t *testing.T, p *Tensor, grad []float32) {
	t.Helper()
	if grad == nil {
		return
	}
	if p.Device == tensor.CPU {
		p.Grad = append([]float32(nil), grad...)
		return
	}
	if err := p.ensureGradGPU().CopyFromHost(floatBytes(grad)); err != nil {
		t.Fatal(err)
	}
}
func TestSGDPyTorchParityAndState(t *testing.T) {
	fixtures := sgdFixtures(t)
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, f := range fixtures {
			t.Run(f.Name, func(t *testing.T) {
				pool2dContext(t, device)
				params := make([]Parameter, len(f.Initial))
				for i, v := range f.Initial {
					name := "weight"
					if i == 1 {
						name = "bias"
					}
					params[i] = Parameter{name, conv2dTestTensor(t, v, []int{len(v)}, device, true)}
				}
				o, err := NewSGD(params, f.Options)
				if err != nil {
					t.Fatal(err)
				}
				defer o.Close()
				EnableGPUProfile(device == tensor.CUDA)
				defer EnableGPUProfile(false)
				for index, step := range f.Steps {
					o.ZeroGrad()
					o.SetLearningRate(step.LR)
					for i, p := range params {
						setSGDGradient(t, p.Value, step.Gradients[i])
					}
					o.Step()
					state, err := o.State()
					if err != nil {
						t.Fatal(err)
					}
					if state.StepCount != index+1 {
						t.Fatal("SGD step counter drifted")
					}
					for i, p := range params {
						conv2dClose(t, "SGD parameter", conv2dHost(t, p.Value, false), step.Values[i], 5e-7, 3e-6)
						if state.Initialized[i] != (step.Buffers[i] != nil) {
							t.Fatal("SGD initialized missing-gradient state early")
						}
						conv2dClose(t, "SGD momentum", state.Buffers[i], step.Buffers[i], 5e-7, 3e-6)
					}
					if index == 0 {
						// Restore into a stateless configuration; saved options and
						// lazy initialization flags must replace constructor defaults.
						path := filepath.Join(t.TempDir(), "sgd.safetensors")
						if err := o.SaveSafeTensors(path); err != nil {
							t.Fatal(err)
						}
						restored, err := NewSGD(params, SGDOptions{LR: .7})
						if err != nil {
							t.Fatal(err)
						}
						if err := restored.LoadSafeTensors(path); err != nil {
							t.Fatal(err)
						}
						loaded, err := restored.State()
						if err != nil {
							t.Fatal(err)
						}
						if !reflect.DeepEqual(state, loaded) {
							t.Fatal("SGD state roundtrip changed values/options")
						}
						restored.Close()
					}
				}
				if device == tensor.CUDA && GPUProfile()["sgd_update"].Count == 0 {
					t.Fatal("SGD left CUDA")
				}
			})
		}
	})
}

func TestSGDCheckpointResumeAndLegacyAdamW(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		newRun := func() (*Module, *SGD, *OneCycle) {
			p := conv2dTestTensor(t, []float32{.3, -.7}, []int{2}, device, true)
			m := &Module{Parameters: []Parameter{{"weight", p}}}
			o, err := NewSGD(m.NamedParameters(), SGDOptions{LR: .03, Momentum: .9, Nesterov: true, WeightDecay: .01})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(o.Close)
			return m, o, NewOneCycle(.03, 8, .3)
		}
		m, o, schedule := newRun()
		o.SetLearningRate(float32(schedule.LR()))
		step := func(m *Module, o *SGD, s *OneCycle) {
			o.ZeroGrad()
			loss := Sum(Mul(m.Parameters[0].Value, m.Parameters[0].Value), 0)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			loss.ReleaseGraph()
			o.Step()
			s.Step(o)
		}
		for i := 0; i < 3; i++ {
			step(m, o, schedule)
		}
		path := filepath.Join(t.TempDir(), "training.safetensors")
		if err := SaveTrainingCheckpoint(path, m, o, schedule, map[string]string{"sampler": "example"}); err != nil {
			t.Fatal(err)
		}
		n, restored, c := newRun()
		meta, err := LoadTrainingCheckpoint(path, n, restored, c)
		if err != nil || meta["sampler"] != "example" {
			t.Fatalf("SGD checkpoint restore: %v", err)
		}
		for i := 3; i < 8; i++ {
			step(m, o, schedule)
			step(n, restored, c)
		}
		left, _ := o.State()
		right, _ := restored.State()
		if !reflect.DeepEqual(m.StateDict(), n.StateDict()) || !reflect.DeepEqual(left, right) || !reflect.DeepEqual(schedule, c) {
			t.Fatal("SGD resume diverged")
		}
		// Wrong optimizer kind must reject before model values are copied.
		adam := NewAdamW(n.NamedParameters(), .1, 0)
		defer adam.Close()
		before := n.StateDict()
		if _, err := LoadTrainingCheckpoint(path, n, adam, c); err == nil || !reflect.DeepEqual(before, n.StateDict()) {
			t.Fatal("optimizer kind mismatch mutated model")
		}
		// Remove the new discriminator to emulate an existing training-1 file.
		legacy := filepath.Join(t.TempDir(), "legacy.safetensors")
		adamSchedule := NewOneCycle(.1, 8, .3)
		if err := SaveTrainingCheckpoint(legacy, n, adam, adamSchedule, nil); err != nil {
			t.Fatal(err)
		}
		entries, metadata, err := readSafetensors(legacy)
		if err != nil {
			t.Fatal(err)
		}
		delete(metadata, "optimizer_kind")
		if err := writeSafetensors(legacy, entries, metadata); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadTrainingCheckpoint(legacy, n, adam, adamSchedule); err != nil {
			t.Fatal(err)
		}
	})
}

func TestSGDValidationVersionsAndSnapshot(t *testing.T) {
	p := testParameter(t, []float32{1, 2})
	defer p.Value.Close()
	for _, options := range []SGDOptions{{LR: -1}, {LR: float32(math.NaN())}, {Momentum: -1}, {Nesterov: true}, {Momentum: .9, Dampening: .2, Nesterov: true}} {
		if _, err := NewSGD([]Parameter{p}, options); err == nil {
			t.Fatal("invalid SGD options accepted")
		}
	}
	if _, err := NewSGD([]Parameter{p, p}, SGDOptions{}); err == nil {
		t.Fatal("duplicate SGD parameters accepted")
	}
	o, err := NewSGD([]Parameter{p}, SGDOptions{LR: .1, Momentum: .9})
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	p.Value.Grad = []float32{.2, -.3}
	o.Step()
	before, _ := o.State()
	bad, _ := o.State()
	bad.Options.Nesterov = true
	bad.Options.Dampening = .5
	if err := o.LoadState(bad); err == nil {
		t.Fatal("invalid restored configuration accepted")
	}
	after, _ := o.State()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("invalid restore modified state")
	}
	bad, _ = o.State()
	bad.Buffers[0][0] = float32(math.NaN())
	if err := o.LoadState(bad); err == nil {
		t.Fatal("invalid momentum accepted")
	}
	snapshot, err := o.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	snapshot.SGD.Buffers[0][0] = 99
	after, _ = o.State()
	if after.Buffers[0][0] == 99 {
		t.Fatal("snapshot aliases live momentum")
	}
	loss := Sum(Mul(p.Value, p.Value), 0)
	defer loss.ReleaseGraph()
	o.Step()
	if err := loss.Backward(); err == nil {
		t.Fatal("SGD mutation did not invalidate saved graph")
	}
	o.Close()
	o.Close()
	if _, err := o.State(); err == nil {
		t.Fatal("closed SGD state accepted")
	}
	a, b := testParameter(t, []float32{1, 2}), testParameter(t, []float32{3, 4})
	a.Name, b.Name = "first", "second"
	defer a.Value.Close()
	defer b.Value.Close()
	invalid, _ := NewSGD([]Parameter{a, b}, SGDOptions{LR: .1})
	defer invalid.Close()
	a.Value.Grad = []float32{1, 1}
	b.Value.Grad = []float32{1}
	panicked := false
	func() { defer func() { panicked = recover() != nil }(); invalid.Step() }()
	if !panicked || invalid.StepCount != 0 || a.Value.Data[0] != 1 {
		t.Fatal("invalid gradient partially updated SGD parameters")
	}
}

func TestSGDCUDACaptureAndBF16Invalidation(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	pool2dContext(t, tensor.CUDA)
	initial := []float32{.3, -.7, .2, .4}
	params := func(name string) Parameter {
		return Parameter{name, conv2dTestTensor(t, initial, []int{2, 2}, tensor.CUDA, true)}
	}
	p, q, warm := params("weight"), params("weight"), params("warm")
	options := SGDOptions{LR: .03, Momentum: .9, Dampening: .2, WeightDecay: .1}
	o, _ := NewSGD([]Parameter{p}, options)
	defer o.Close()
	eager, _ := NewSGD([]Parameter{q}, options)
	defer eager.Close()
	precompile, _ := NewSGD([]Parameter{warm}, options)
	defer precompile.Close()
	setSGDGradient(t, warm.Value, []float32{.2, -.3, .1, .4})
	precompile.Step()
	setSGDGradient(t, p.Value, []float32{.2, -.3, .1, .4})
	if err := o.PrepareGraph(); err != nil {
		t.Fatal(err)
	}
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	graph, err := cuda.Capture(stream, func() error { o.Step(); return nil })
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	for i := 0; i < 3; i++ {
		if err := graph.Launch(); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		eager.ZeroGrad()
		setSGDGradient(t, q.Value, []float32{.2, -.3, .1, .4})
		eager.Step()
	}
	left, err := o.State()
	if err != nil {
		t.Fatal(err)
	}
	right, _ := eager.State()
	if left.StepCount != 3 {
		t.Fatal("captured SGD step counter did not advance")
	}
	conv2dClose(t, "captured SGD weights", conv2dHost(t, p.Value, false), conv2dHost(t, q.Value, false), 1e-6, 1e-6)
	conv2dClose(t, "captured SGD momentum", left.Buffers[0], right.Buffers[0], 1e-6, 1e-6)
	graph.Close()
	// Eager updates after replay sync the host count and invalidate FP32 aliases.
	p.Value.ensureBF16()
	alias := Reshape(p.Value, 4)
	defer alias.Close()
	o.Step()
	if o.StepCount != 4 || alias.currentBF16() != nil {
		t.Fatal("eager SGD did not sync count/invalidate BF16 alias")
	}
}
