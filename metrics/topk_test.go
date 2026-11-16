package metrics

import (
	"math"
	"testing"
)

func TestTopKAccuracy(t *testing.T) {
	a, err := NewTopKAccuracy(2)
	if err != nil {
		t.Fatal(err)
	}
	// row 0: label 1 is the best score
	// row 1: label 0 is the second best score
	// row 2: label 2 is the worst score
	scores := []float32{
		0.1, 0.9, 0.0,
		0.5, 0.7, 0.2,
		0.3, 0.2, 0.1,
	}
	labels := []int{1, 0, 2}
	if err := a.UpdateLogits(scores, labels, 3, -1); err != nil {
		t.Fatal(err)
	}
	if a.Count() != 3 || a.Correct() != 2 {
		t.Fatalf("count=%d correct=%d, want 3/2", a.Count(), a.Correct())
	}
	if got := a.Value(); math.Abs(got-2.0/3) > 1e-12 {
		t.Fatalf("value = %g, want %g", got, 2.0/3)
	}
}

func TestTopKAccuracyK1MatchesAccuracy(t *testing.T) {
	scores := []float32{
		0.1, 0.9, 0.0,
		0.5, 0.7, 0.2,
		0.1, 0.2, 0.3,
		0.4, 0.4, 0.1,
	}
	labels := []int{1, 0, 2, 2}
	var top1 Accuracy
	if err := top1.UpdateLogits(scores, labels, 3, -1); err != nil {
		t.Fatal(err)
	}
	topK, err := NewTopKAccuracy(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := topK.UpdateLogits(scores, labels, 3, -1); err != nil {
		t.Fatal(err)
	}
	if top1.Correct() != topK.Correct() || top1.Count() != topK.Count() {
		t.Fatalf("top1 %d/%d, topK %d/%d", top1.Correct(), top1.Count(), topK.Correct(), topK.Count())
	}
}

func TestTopKAccuracyTiesAtBoundary(t *testing.T) {
	a, err := NewTopKAccuracy(2)
	if err != nil {
		t.Fatal(err)
	}
	// label 0 ties with class 1 for the best score, so it is within the top 2.
	if err := a.UpdateLogits([]float32{0.5, 0.5, 0.1}, []int{0}, 3, -1); err != nil {
		t.Fatal(err)
	}
	if a.Correct() != 1 {
		t.Fatalf("correct = %d, want 1", a.Correct())
	}
}

func TestTopKAccuracyIgnoresIndex(t *testing.T) {
	a, err := NewTopKAccuracy(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.UpdateLogits([]float32{0.1, 0.9, 0.0, 0.2}, []int{0, 2}, 2, 2); err != nil {
		t.Fatal(err)
	}
	if a.Count() != 1 || a.Correct() != 0 {
		t.Fatalf("count=%d correct=%d, want 1/0", a.Count(), a.Correct())
	}
}

func TestTopKAccuracyRejectsInvalid(t *testing.T) {
	if _, err := NewTopKAccuracy(0); err == nil {
		t.Fatal("expected an error for k=0")
	}
	a, err := NewTopKAccuracy(2)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name            string
		scores          []float32
		labels          []int
		classes, ignore int
	}{
		{"zero classes", []float32{0}, []int{0}, 0, -1},
		{"shape mismatch", []float32{0, 1, 2}, []int{0}, 2, -1},
		{"label out of range", []float32{0, 1}, []int{5}, 2, -1},
		{"non-finite", []float32{0, float32(math.Inf(1))}, []int{0}, 2, -1},
	} {
		if err := a.UpdateLogits(tc.scores, tc.labels, tc.classes, tc.ignore); err == nil {
			t.Fatalf("%s: expected an error", tc.name)
		}
		if a.Count() != 0 {
			t.Fatalf("%s: accumulator changed after an error", tc.name)
		}
	}
}

func TestTopKAccuracyReset(t *testing.T) {
	a, err := NewTopKAccuracy(3)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.UpdateLogits([]float32{0.1, 0.9}, []int{0}, 2, -1); err != nil {
		t.Fatal(err)
	}
	a.Reset()
	if a.Count() != 0 || a.Correct() != 0 || a.K() != 3 {
		t.Fatalf("reset left count=%d correct=%d k=%d", a.Count(), a.Correct(), a.K())
	}
}
