package data

import (
	"image"
	"math"
	"testing"
)

// TestAuditStandardizerApplyBatchRejectsShortValues checks that a batch whose
// value count does not match its shape is rejected instead of panicking.
func TestAuditStandardizerApplyBatchRejectsShortValues(t *testing.T) {
	standardizer := Standardizer{Mean: []float32{0, 0}, Std: []float32{1, 1}}
	if err := standardizer.ApplyBatch(&RegressionBatch{Values: []float32{1, 2}, Shape: []int{4, 2}}); err == nil {
		t.Fatal("short batch was accepted")
	}
	if err := standardizer.ApplyBatch(&RegressionBatch{Values: []float32{1, 2, 3}, Shape: []int{1, 2}}); err == nil {
		t.Fatal("long batch was accepted")
	}
	if err := standardizer.ApplyBatch(&RegressionBatch{Values: []float32{}, Shape: []int{0, 2}}); err != nil {
		t.Fatalf("empty batch was rejected: %v", err)
	}
}

// TestAuditStandardizerKeepsScaleForLargeMean checks that a feature with a
// large mean keeps its spread: the one-pass variance formula loses it.
func TestAuditStandardizerKeepsScaleForLargeMean(t *testing.T) {
	samples := []RegressionSample{
		{Features: []float32{1e9 - 1}, Target: 0},
		{Features: []float32{1e9}, Target: 0},
		{Features: []float32{1e9 + 1}, Target: 0},
	}
	standardizer, err := FitStandardizer(samples)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(standardizer.Std[0])-1) > 1e-2 {
		t.Fatalf("standard deviation %v, want about 1", standardizer.Std[0])
	}
	applied := standardizer.Apply(samples[1])
	if math.Abs(float64(applied.Features[0])) > 1e-2 {
		t.Fatalf("standardized value %v, want about 0", applied.Features[0])
	}
}

// TestAuditScaleImageRejectsNonPositiveSize checks that a zero or negative
// size returns no pixels instead of dividing by zero.
func TestAuditScaleImageRejectsNonPositiveSize(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 4, 4))
	if got := ScaleImage(img, 0, 4, 1); got != nil {
		t.Fatalf("zero width returned %d pixels", len(got))
	}
	if got := ScaleImage(img, 4, -1, 3); got != nil {
		t.Fatalf("negative height returned %d pixels", len(got))
	}
	if got := ScaleImage(img, 2, 2, 1); len(got) != 4 {
		t.Fatalf("scaled image has %d pixels, want 4", len(got))
	}
}

// TestAuditSamplerEpochAccessor checks that the epoch accessor reports the
// same epoch as the state without copying the permutation.
func TestAuditSamplerEpochAccessor(t *testing.T) {
	sampler, err := NewIndexSampler(8, SamplerOptions{Shuffle: true, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if sampler.Epoch() != sampler.State().Epoch {
		t.Fatalf("epoch %d != state epoch %d", sampler.Epoch(), sampler.State().Epoch)
	}
	if err := sampler.NextEpoch(); err != nil {
		t.Fatal(err)
	}
	if sampler.Epoch() != sampler.State().Epoch {
		t.Fatalf("epoch %d != state epoch %d", sampler.Epoch(), sampler.State().Epoch)
	}
}
