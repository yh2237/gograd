package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeDataset(t *testing.T, dir string, rows, features int, seed uint64) string {
	t.Helper()
	path := filepath.Join(dir, "data.csv")
	if err := generateDataset(path, rows, features, seed); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestGenerateDatasetIsDeterministic(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "a.csv")
	second := filepath.Join(dir, "b.csv")
	if err := generateDataset(first, 50, 3, 9); err != nil {
		t.Fatal(err)
	}
	if err := generateDataset(second, 50, 3, 9); err != nil {
		t.Fatal(err)
	}
	a, err := os.ReadFile(first)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatal("generated datasets differ")
	}
	if err := generateDataset(filepath.Join(dir, "c.csv"), 50, 3, 10); err != nil {
		t.Fatal(err)
	}
	c, err := os.ReadFile(filepath.Join(dir, "c.csv"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(a, c) {
		t.Fatal("seed did not change the dataset")
	}
	if err := generateDataset(filepath.Join(dir, "d.csv"), 1, 3, 9); err == nil {
		t.Fatal("invalid shape accepted")
	}
}

func TestTabularMidEpochResume(t *testing.T) {
	dir := t.TempDir()
	dataPath := writeDataset(t, dir, 400, 4, 5)
	base := trainConfig{Data: dataPath, Device: "cpu", Steps: 12, Batch: 16, Hidden: 8, Seed: 3}
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
	partial, err := train(context.Background(), part, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if partial.Steps != 8 {
		t.Fatalf("partial steps %d", partial.Steps)
	}
	resume := base
	resume.Out = part.Out
	resume.Resume = part.Out + ".training.safetensors"
	resume.Workers, resume.Prefetch = 2, 2
	restored, err := train(context.Background(), resume, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Steps != whole.Steps || restored.ValidRMSE != whole.ValidRMSE {
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
	// Changing an identity field must be rejected before loading live state.
	for name, mutate := range map[string]func(*trainConfig){
		"batch":        func(c *trainConfig) { c.Batch = 8 },
		"steps":        func(c *trainConfig) { c.Steps = 20 },
		"seed":         func(c *trainConfig) { c.Seed = 4 },
		"hidden":       func(c *trainConfig) { c.Hidden = 16 },
		"optimizer":    func(c *trainConfig) { c.Optimizer = "sgd" },
		"learningrate": func(c *trainConfig) { c.LearningRate = .02 },
	} {
		changed := resume
		changed.Out = filepath.Join(dir, "reject-"+name+".safetensors")
		mutate(&changed)
		if _, err := train(context.Background(), changed, io.Discard); err == nil {
			t.Fatalf("%s change accepted", name)
		}
	}
}

func TestTabularDataChangeIsRejected(t *testing.T) {
	dir := t.TempDir()
	dataPath := writeDataset(t, dir, 200, 3, 11)
	base := trainConfig{Data: dataPath, Device: "cpu", Steps: 8, Batch: 16, Hidden: 6, Seed: 2}
	first := base
	first.Out = filepath.Join(dir, "first.safetensors")
	first.StopAfter = 4
	if _, err := train(context.Background(), first, io.Discard); err != nil {
		t.Fatal(err)
	}
	// Rewriting the file changes its fingerprint even when the shape matches.
	raw, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	fields := strings.Split(lines[1], ",")
	fields[0] = "9999"
	lines[1] = strings.Join(fields, ",")
	if err := os.WriteFile(dataPath, []byte(strings.Join(lines, "\n")+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	changed := base
	changed.Out = filepath.Join(dir, "second.safetensors")
	changed.Resume = first.Out + ".training.safetensors"
	if _, err := train(context.Background(), changed, io.Discard); err == nil {
		t.Fatal("changed data accepted on resume")
	}
}

func TestTabularValidation(t *testing.T) {
	dir := t.TempDir()
	dataPath := writeDataset(t, dir, 100, 3, 1)
	valid := trainConfig{Data: dataPath, Device: "cpu", Steps: 4, Batch: 8, Hidden: 4, Out: filepath.Join(dir, "ok.safetensors")}
	if _, err := train(context.Background(), valid, io.Discard); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*trainConfig){
		"optimizer":   func(c *trainConfig) { c.Optimizer = "rmsprop" },
		"device":      func(c *trainConfig) { c.Device = "tpu" },
		"hidden":      func(c *trainConfig) { c.Hidden = 0 },
		"workers":     func(c *trainConfig) { c.Prefetch = 2 },
		"valid-count": func(c *trainConfig) { c.ValidCount = 100 },
		"samepath":    func(c *trainConfig) { c.Checkpoint = c.Out },
		"negative-lr": func(c *trainConfig) { c.LearningRate = -1 },
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
}
