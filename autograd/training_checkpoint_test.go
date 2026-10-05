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

func TestTrainingCheckpointResume(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { ctx.Close() })
			}
			newRun := func() (*Module, *AdamW, *OneCycle, *WindowSampler) {
				p, err := New([]float32{.3, -.7}, []int{1, 1, 2}, device, true)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { p.Close() })
				m := &Module{Parameters: []Parameter{{Name: "weight", Value: p}}}
				o := NewAdamW(m.NamedParameters(), .01, .02)
				t.Cleanup(o.Close)
				c := NewOneCycle(.01, 8, .3)
				o.LR = float32(c.LR())
				s, err := NewWindowSampler([]int{5, 8}, 4, 1, .15, 7)
				if err != nil {
					t.Fatal(err)
				}
				return m, o, c, s
			}
			m, opt, schedule, sampler := newRun()
			step := func(module *Module, optimizer *AdamW, c *OneCycle, s *WindowSampler) float32 {
				w := s.Batch()[0]
				x, err := New([]float32{float32(w.Index + 1), float32(w.Start + 1)}, []int{1, 1, 2}, device, false)
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				target, err := Zeros([]int{1, 1, 2}, device, false)
				if err != nil {
					t.Fatal(err)
				}
				defer target.Close()
				optimizer.ZeroGrad()
				loss := MaskedLoss(Mul(module.Parameters[0].Value, x), target, true)
				defer loss.ReleaseGraph()
				values, err := loss.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				optimizer.Step()
				c.Step(optimizer)
				return values[0]
			}
			for i := 0; i < 3; i++ {
				step(m, opt, schedule, sampler)
			}
			data, err := json.Marshal(sampler.State())
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "training.safetensors")
			if err := SaveTrainingCheckpoint(path, m, opt, schedule, map[string]string{"sampler": string(data)}); err != nil {
				t.Fatal(err)
			}
			resumedModel, resumedOpt, resumedSchedule, resumedSampler := newRun()
			// Hyperparameters must be restored, not inherited from new defaults.
			resumedOpt.LR, resumedOpt.WeightDecay, resumedOpt.Beta1 = .9, .7, .8
			metadata, err := LoadTrainingCheckpoint(path, resumedModel, resumedOpt, resumedSchedule)
			if err != nil {
				t.Fatal(err)
			}
			var sampleState WindowSamplerState
			if err := json.Unmarshal([]byte(metadata["sampler"]), &sampleState); err != nil {
				t.Fatal(err)
			}
			if err := resumedSampler.LoadState(sampleState); err != nil {
				t.Fatal(err)
			}
			for i := 3; i < 8; i++ {
				want := step(m, opt, schedule, sampler)
				got := step(resumedModel, resumedOpt, resumedSchedule, resumedSampler)
				if math.Abs(float64(want-got)) > 1e-7 {
					t.Fatalf("loss at step %d: %g != %g", i, got, want)
				}
			}
			if !reflect.DeepEqual(m.StateDict(), resumedModel.StateDict()) {
				t.Fatal("resumed model diverged")
			}
			wantState, _ := opt.State()
			gotState, _ := resumedOpt.State()
			if !reflect.DeepEqual(wantState, gotState) || !reflect.DeepEqual(schedule, resumedSchedule) || !reflect.DeepEqual(sampler.State(), resumedSampler.State()) {
				t.Fatal("resumed optimizer/scheduler/sampler diverged")
			}
			// Atomic replacement of an existing checkpoint must also work.
			if err := SaveTrainingCheckpoint(path, m, opt, schedule, nil); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestOptimizerRejectsStateBeforeMutation(t *testing.T) {
	a, b := testParameter(t, []float32{1, 2}), testParameter(t, []float32{3, 4})
	a.Name, b.Name = "first", "second"
	opt := NewAdamW([]Parameter{a, b}, .01, .1)
	state, err := opt.State()
	if err != nil {
		t.Fatal(err)
	}
	state.Moments[0].M[0] = 5
	state.Shapes[1] = []int{1, 2}
	if err := opt.LoadState(state); err == nil {
		t.Fatal("accepted different parameter shape")
	}
	if opt.M[0][0] != 0 {
		t.Fatal("invalid state partially modified optimizer")
	}
	state.Shapes[1] = []int{2}
	state.StepCount = -1
	if err := opt.LoadState(state); err == nil {
		t.Fatal("accepted negative step")
	}
}

func TestSamplerStateValidationAndCoverage(t *testing.T) {
	s, err := NewWindowSampler([]int{1, 6}, 4, 12, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	s.Batch()
	state := s.State()
	clone, _ := NewWindowSampler([]int{1, 6}, 4, 12, 1, 99)
	if err := clone.LoadState(state); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.Batch(), clone.Batch()) {
		t.Fatal("sampler stream did not resume")
	}
	seenLast := false
	for i := 0; i < 100; i++ {
		for _, window := range s.Batch() {
			if !window.DropContext || window.Start < 0 || (window.Index == 0 && window.Start != 0) || window.Start > 2 {
				t.Fatal("invalid sampled window")
			}
			seenLast = seenLast || window.Index == 1 && window.Start == 2
		}
	}
	if !seenLast {
		t.Fatal("final complete window was unreachable")
	}
	state.Frames[0]++
	before := s.State()
	if err := s.LoadState(state); err == nil {
		t.Fatal("accepted different sequence lengths")
	}
	if !reflect.DeepEqual(before, s.State()) {
		t.Fatal("invalid sampler state changed stream")
	}
	l := NewIndexLoader(9, 4, 1)
	l.Epoch()
	indexState := l.State()
	other := NewIndexLoader(9, 4, 1)
	if err := other.LoadState(indexState); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(l.Epoch(), other.Epoch()) {
		t.Fatal("index loader did not resume epoch")
	}
}

func TestRecursiveTrainingMode(t *testing.T) {
	dropout := &DropoutLayer{Training: true}
	leaf := &Module{OnTrainingChange: dropout.Train}
	root := &Module{Children: []NamedModule{{Name: "branch", Module: &Module{Children: []NamedModule{{Name: "leaf", Module: leaf}}}}}}
	SetTraining(false, root)
	if root.Training || leaf.Training || dropout.Training {
		t.Fatal("eval did not reach nested dropout")
	}
	SetTraining(true, root)
	if !root.Training || !leaf.Training || !dropout.Training {
		t.Fatal("train did not reach nested dropout")
	}
}

func TestSafeTensorsRejectsOverflowAndOverlap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.safetensors")
	header := map[string]any{
		"first":  safeHeader{"F32", []int{1}, [2]int{0, 4}},
		"second": safeHeader{"F32", []int{1}, [2]int{0, 4}},
	}
	write := func() {
		data, _ := json.Marshal(header)
		file := make([]byte, 8)
		for i := 0; i < 8; i++ {
			file[i] = byte(uint64(len(data)) >> (8 * i))
		}
		file = append(append(file, data...), make([]byte, 4)...)
		if err := os.WriteFile(path, file, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write()
	if _, _, err := readSafetensors(path); err == nil {
		t.Fatal("accepted overlapping tensor ranges")
	}
	header["first"] = safeHeader{"F32", []int{math.MaxInt, 4}, [2]int{0, 4}}
	write()
	if _, _, err := readSafetensors(path); err == nil {
		t.Fatal("accepted overflowing shape")
	}
}

func TestTrainingCheckpointRejectsBeforeCopy(t *testing.T) {
	p := testParameter(t, []float32{1, 2})
	module := &Module{Parameters: []Parameter{p}}
	opt := NewAdamW(module.NamedParameters(), .01, .1)
	schedule := NewOneCycle(.01, 8, .3)
	opt.LR = float32(schedule.LR())
	path := filepath.Join(t.TempDir(), "training.safetensors")
	if err := SaveTrainingCheckpoint(path, module, opt, schedule, nil); err != nil {
		t.Fatal(err)
	}
	entries, metadata, err := readSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	entry := entries["model.weight"]
	entry.Values[0] = 99
	entries["model.weight"] = entry
	var state OptimizerState
	json.Unmarshal([]byte(metadata["optimizer"]), &state)
	state.Beta1 = 2
	data, _ := json.Marshal(state)
	metadata["optimizer"] = string(data)
	if err := writeSafetensors(path, entries, metadata); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTrainingCheckpoint(path, module, opt, schedule); err == nil {
		t.Fatal("accepted invalid optimizer configuration")
	}
	if p.Value.Data[0] != 1 || opt.Beta1 != .9 || schedule.StepCount != 0 {
		t.Fatal("invalid checkpoint changed live state")
	}
	before, _ := os.ReadFile(path)
	if err := writeSafetensors(path, map[string]safeTensorEntry{"bad": {Shape: []int{10}, Values: []float32{1}}}, nil); err == nil {
		t.Fatal("accepted mismatched save shape")
	}
	after, _ := os.ReadFile(path)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("failed save replaced existing checkpoint")
	}
}

func TestOneCycleSinglePointWarmup(t *testing.T) {
	c := NewOneCycle(.002, 10, .1)
	for i := 0; i <= c.Total; i++ {
		c.StepCount = i
		lr := c.LR()
		if math.IsNaN(lr) || math.IsInf(lr, 0) {
			t.Fatalf("non-finite LR at step %d", i)
		}
	}
}
