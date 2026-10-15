package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	randv2 "math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeWAVFixture(t *testing.T, dir string, seconds float64, sampleRate int, seed uint64) string {
	t.Helper()
	path := filepath.Join(dir, "tone.wav")
	if err := writeGeneratedWAV(path, int(seconds*float64(sampleRate)), sampleRate, seed); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestNoisyWindowsAreOrderIndependent(t *testing.T) {
	clean := frames(generateSignal(4096, 16000, 3), 64)
	dataset := &noisyWindows{clean: clean, noise: .2, seed: 5}
	first, err := dataset.Get(context.Background(), 7)
	if err != nil {
		t.Fatal(err)
	}
	// Reading other windows first must not change window 7.
	if _, err := dataset.Get(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	second, err := dataset.Get(context.Background(), 7)
	if err != nil {
		t.Fatal(err)
	}
	if !equalFloat32(first.Input, second.Input) || !equalFloat32(first.Target, second.Target) {
		t.Fatal("augmentation depended on access order")
	}
	// The target is the clean frame; the input differs by the added noise.
	if !equalFloat32(first.Target, clean[7]) || equalFloat32(first.Input, first.Target) {
		t.Fatal("augmentation did not keep the clean target")
	}
	// Preserve the exact formula used before extracting the common helper.
	rng := randv2.New(randv2.NewPCG(dataset.seed, uint64(7)+0x9e3779b97f4a7c15))
	for i, value := range clean[7] {
		want := value + float32(rng.NormFloat64())*dataset.noise
		if first.Input[i] != want {
			t.Fatalf("shared helper changed input[%d]: %g != %g", i, first.Input[i], want)
		}
	}
	// Returned inputs and targets must not alias the stored clean frame.
	first.Input[0], first.Target[0] = 99, 99
	again, err := dataset.Get(context.Background(), 7)
	if err != nil || !equalFloat32(again.Input, second.Input) || !equalFloat32(again.Target, second.Target) {
		t.Fatal("caller mutation changed future augmentation", err)
	}
	// A different seed (checkpoint identity) changes the noise.
	other := &noisyWindows{clean: clean, noise: .2, seed: 6}
	third, err := other.Get(context.Background(), 7)
	if err != nil {
		t.Fatal(err)
	}
	if equalFloat32(second.Input, third.Input) {
		t.Fatal("seed did not change the noise")
	}
	if _, err := dataset.Get(nil, 0); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := dataset.Get(context.Background(), len(clean)); err == nil {
		t.Fatal("out-of-range window accepted")
	}
}

func TestCollateWindows(t *testing.T) {
	batch, err := collateWindows([]windowSample{{Input: []float32{1, 2}, Target: []float32{3, 4}}, {Input: []float32{5, 6}, Target: []float32{7, 8}}})
	if err != nil {
		t.Fatal(err)
	}
	if batch.Shape[0] != 2 || batch.Shape[1] != 2 || batch.Inputs[3] != 6 || batch.Targets[2] != 7 {
		t.Fatalf("batch %+v", batch)
	}
	if _, err := collateWindows(nil); err == nil {
		t.Fatal("empty batch accepted")
	}
	if _, err := collateWindows([]windowSample{{Input: []float32{1}, Target: []float32{2}}, {Input: []float32{1}, Target: []float32{2, 3}}}); err == nil {
		t.Fatal("unequal windows accepted")
	}
}

func TestAudioMidEpochResume(t *testing.T) {
	dir := t.TempDir()
	wavPath := writeWAVFixture(t, dir, .5, 8000, 9)
	base := trainConfig{WAV: wavPath, Device: "cpu", Steps: 12, Batch: 16, Hidden: 8, Window: 32, Seed: 3, Noise: .2}
	full := base
	full.Out = filepath.Join(dir, "full.safetensors")
	whole, err := train(context.Background(), full, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	part := base
	part.Out = filepath.Join(dir, "part.safetensors")
	part.StopAfter = 8
	part.Workers, part.Prefetch = 3, 3
	if _, err := train(context.Background(), part, io.Discard); err != nil {
		t.Fatal(err)
	}
	resume := base
	resume.Out = part.Out
	resume.Resume = part.Out + ".training.safetensors"
	resume.Workers, resume.Prefetch = 2, 2
	restored, err := train(context.Background(), resume, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Steps != whole.Steps || restored.InputSNR != whole.InputSNR || restored.OutputSNR != whole.OutputSNR {
		t.Fatalf("resumed stats %+v != %+v", restored, whole)
	}
	wholeWeights, err := os.ReadFile(full.Out)
	if err != nil {
		t.Fatal(err)
	}
	restoredWeights, err := os.ReadFile(part.Out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wholeWeights, restoredWeights) {
		t.Fatal("resumed weights differ from the uninterrupted run")
	}
	for name, mutate := range map[string]func(*trainConfig){
		"noise":     func(c *trainConfig) { c.Noise = .3 },
		"window":    func(c *trainConfig) { c.Window = 16 },
		"batch":     func(c *trainConfig) { c.Batch = 8 },
		"seed":      func(c *trainConfig) { c.Seed = 4 },
		"hidden":    func(c *trainConfig) { c.Hidden = 16 },
		"optimizer": func(c *trainConfig) { c.Optimizer = "sgd" },
	} {
		changed := resume
		changed.Out = filepath.Join(dir, "reject-"+name+".safetensors")
		mutate(&changed)
		if _, err := train(context.Background(), changed, io.Discard); err == nil {
			t.Fatalf("%s change accepted", name)
		}
	}
}

func TestAudioDataChangeIsRejected(t *testing.T) {
	dir := t.TempDir()
	wavPath := writeWAVFixture(t, dir, .4, 8000, 2)
	base := trainConfig{WAV: wavPath, Device: "cpu", Steps: 6, Batch: 8, Hidden: 4, Window: 32, Seed: 1}
	first := base
	first.Out = filepath.Join(dir, "first.safetensors")
	first.StopAfter = 3
	if _, err := train(context.Background(), first, io.Discard); err != nil {
		t.Fatal(err)
	}
	if err := writeGeneratedWAV(wavPath, 3200, 8000, 3); err != nil {
		t.Fatal(err)
	}
	changed := base
	changed.Out = filepath.Join(dir, "second.safetensors")
	changed.Resume = first.Out + ".training.safetensors"
	if _, err := train(context.Background(), changed, io.Discard); err == nil {
		t.Fatal("changed audio accepted on resume")
	}
}

func TestAudioValidation(t *testing.T) {
	dir := t.TempDir()
	wavPath := writeWAVFixture(t, dir, .3, 8000, 1)
	valid := trainConfig{WAV: wavPath, Device: "cpu", Steps: 4, Batch: 8, Hidden: 4, Window: 32, Out: filepath.Join(dir, "ok.safetensors")}
	if _, err := train(context.Background(), valid, io.Discard); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*trainConfig){
		"optimizer": func(c *trainConfig) { c.Optimizer = "lion" },
		"device":    func(c *trainConfig) { c.Device = "tpu" },
		"window":    func(c *trainConfig) { c.Window = 0 },
		"hidden":    func(c *trainConfig) { c.Hidden = 0 },
		"workers":   func(c *trainConfig) { c.Prefetch = 2 },
		"noise":     func(c *trainConfig) { c.Noise = -1 },
		"valid":     func(c *trainConfig) { c.ValidCount = 1000 },
		"samepath":  func(c *trainConfig) { c.Checkpoint = c.Out },
	} {
		bad := valid
		bad.Out = filepath.Join(dir, "bad-"+name+".safetensors")
		mutate(&bad)
		if _, err := train(context.Background(), bad, io.Discard); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if _, err := train(nil, valid, io.Discard); err == nil {
		t.Fatal("nil context accepted")
	}
	// A WAV shorter than two windows is rejected rather than trained on.
	short := valid
	short.Out = filepath.Join(dir, "short.safetensors")
	short.WAV = writeWAVFixture(t, filepath.Join(dir, "short"), .004, 8000, 1)
	if _, err := train(context.Background(), short, io.Discard); err == nil {
		t.Fatal("short WAV accepted")
	}
}

func equalFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type cancelOnProgress struct{ cancel context.CancelFunc }

func (w cancelOnProgress) Write(p []byte) (int, error) {
	w.cancel()
	return len(p), nil
}

func TestAudioCancelSavesCheckpoint(t *testing.T) {
	for _, beforeUpdate := range []bool{false, true} {
		t.Run(map[bool]string{false: "after-update", true: "deadline-before-update"}[beforeUpdate], func(t *testing.T) {
			dir := t.TempDir()
			base := trainConfig{WAV: writeWAVFixture(t, dir, .3, 8000, 9), Device: "cpu", Steps: 6, Batch: 8, Hidden: 4, Window: 32, Seed: 3, Noise: .2}
			full := base
			full.Out = filepath.Join(dir, "full.safetensors")
			if _, err := train(context.Background(), full, io.Discard); err != nil {
				t.Fatal(err)
			}
			part := base
			part.Out = filepath.Join(dir, "part.safetensors")
			part.Workers, part.Prefetch = 2, 3
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			wantErr, wantSteps := context.Canceled, 1
			if beforeUpdate {
				ctx, cancel = context.WithDeadline(context.Background(), time.Unix(0, 0))
				defer cancel()
				wantErr, wantSteps = context.DeadlineExceeded, 0
			}
			stats, err := train(ctx, part, cancelOnProgress{cancel})
			if !errors.Is(err, wantErr) || stats.Steps != wantSteps {
				t.Fatalf("stop stats=%+v err=%v", stats, err)
			}
			if _, err := os.Stat(part.Out); !os.IsNotExist(err) {
				t.Fatal("cancellation wrote inference weights")
			}
			resume := base
			resume.Out, resume.Resume = part.Out, part.Out+".training.safetensors"
			if _, err := train(context.Background(), resume, io.Discard); err != nil {
				t.Fatal(err)
			}
			left, err := os.ReadFile(full.Out + ".training.safetensors")
			if err != nil {
				t.Fatal(err)
			}
			right, err := os.ReadFile(resume.Out + ".training.safetensors")
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(left, right) {
				t.Fatal("resumed model/optimizer/scheduler/data state changed")
			}
		})
	}
}
