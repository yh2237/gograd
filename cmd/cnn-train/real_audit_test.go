package main

import (
	"bytes"
	"io"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/tensor"
)

func realAuditFixture(t *testing.T, count int) (string, string) {
	t.Helper()
	dir := t.TempDir()
	images, labels := filepath.Join(dir, "images.idx"), filepath.Join(dir, "labels.idx")
	pixels, targets := make([][]byte, count), make([]byte, count)
	for i := range pixels {
		pixels[i] = bytes.Repeat([]byte{byte(i * 20), 64, 128, 255}, 4)
		targets[i] = byte(i % 2)
	}
	if err := data.WriteIDX(images, labels, 4, 4, pixels, targets); err != nil {
		t.Fatal(err)
	}
	return images, labels
}

func TestRealIOAuditValidationSplitAndReloadCPU(t *testing.T) {
	images, labels := realAuditFixture(t, 5)
	var log bytes.Buffer
	output := filepath.Join(t.TempDir(), "model.safetensors")
	stats, err := trainReal(tensor.CPU, 1, 1, 4, images, labels, output, &log, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(log.String(), "samples=train1/valid4") || !strings.Contains(log.String(), "evaluation=validation") || !strings.Contains(log.String(), "reload_max_abs=") {
		t.Fatalf("missing split/evaluation/reload: %s", log.String())
	}
	if stats.finalLoss <= 0 || math.IsNaN(float64(stats.finalLoss)) || stats.reloadMaxAbs > 1e-5 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestRealIOAuditNoValidationMeasuresTrainingCPU(t *testing.T) {
	images, labels := realAuditFixture(t, 2)
	var log bytes.Buffer
	stats, err := trainReal(tensor.CPU, 1, 0, 0, images, labels, "", &log, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(log.String(), "samples=train2/valid0 evaluation=training") || stats.finalLoss <= 0 || stats.finalLoss == stats.initialLoss {
		t.Fatalf("unmeasured training metrics: %+v %s", stats, log.String())
	}
}

func TestRealIOAuditRejectsConfigurationEarly(t *testing.T) {
	for _, tc := range []struct{ steps, train, valid int }{{1, -1, 4}, {1, 1, -4}, {0, 1, 1}} {
		_, err := trainReal(tensor.CPU, tc.steps, tc.train, tc.valid, "missing-images", "missing-labels", "", io.Discard, false)
		if err == nil || strings.Contains(err.Error(), "IDX") {
			t.Fatalf("configuration %+v did not fail before file access: %v", tc, err)
		}
	}
}

func TestRealIOAuditBatchBound(t *testing.T) {
	images, labels := realAuditFixture(t, 67)
	idx, err := data.OpenIDX(images, labels)
	if err != nil {
		t.Fatal(err)
	}
	ds := idxRange{idx: idx, start: 1, count: 66}
	for start, want := range map[int]int{0: 64, 64: 2} {
		batch, err := realBatch(ds, start)
		if err != nil || len(batch.Labels) != want || batch.Shape[0] != want {
			t.Fatalf("batch start=%d labels=%d shape=%v err=%v", start, len(batch.Labels), batch.Shape, err)
		}
	}
}

func TestRealIOAuditBatchedEvaluationCPU(t *testing.T) {
	images, labels := realAuditFixture(t, 67)
	idx, err := data.OpenIDX(images, labels)
	if err != nil {
		t.Fatal(err)
	}
	m, err := newClassifier(tensor.CPU, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer m.close()
	m.module.Train(false)
	execution := autograd.NewExecutionContext()
	if err := execution.BindModule(&m.module); err != nil {
		t.Fatal(err)
	}
	ds := idxRange{idx: idx, count: idx.Len()}
	loss, accuracy, err := evaluateReal(m, execution, ds, tensor.CPU, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Compare the weighted, bounded evaluation with a single small full batch.
	samples := make([]data.FloatSample, idx.Len())
	for i := range samples {
		samples[i], err = ds.Get(t.Context(), i)
		if err != nil {
			t.Fatal(err)
		}
	}
	batch, err := data.Stack(samples)
	if err != nil {
		t.Fatal(err)
	}
	wantLoss, hits, err := evaluateRealBatch(m, execution, batch, tensor.CPU, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(loss-wantLoss)) > 1e-5 || accuracy != float32(hits)/float32(idx.Len()) {
		t.Fatalf("batched loss/accuracy=%g/%g full=%g/%g", loss, accuracy, wantLoss, float32(hits)/float32(idx.Len()))
	}
}

func TestRealIOAuditReloadCheckDetectsChangedLogitsCPU(t *testing.T) {
	images, labels := realAuditFixture(t, 1)
	idx, err := data.OpenIDX(images, labels)
	if err != nil {
		t.Fatal(err)
	}
	original, err := newClassifier(tensor.CPU, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer original.close()
	loaded, err := newClassifier(tensor.CPU, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.close()
	execution := autograd.NewExecutionContext()
	for _, m := range []*classifier{original, loaded} {
		m.module.Train(false)
		if err := execution.BindModule(&m.module); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range loaded.module.NamedParameters() {
		if p.Name == "head.bias" {
			if err := p.Value.CopyFrom([]float32{100, -100}); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := compareRealInference(original, loaded, execution, idxRange{idx: idx, count: 1}, tensor.CPU); err == nil || !strings.Contains(err.Error(), "changed logits") {
		t.Fatalf("changed inference was not rejected: %v", err)
	}
}
