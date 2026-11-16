package metrics

import (
	"math"
	"testing"
)

func TestRegressionR2(t *testing.T) {
	var r Regression
	prediction := []float32{1, 2, 3, 4}
	target := []float32{1, 2, 3, 5}
	if err := r.Update(prediction, target); err != nil {
		t.Fatal(err)
	}
	// SS_res = 1, mean = 2.75, SS_tot = 8.75
	want := 1 - 1/8.75
	if got := r.R2(); math.Abs(got-want) > 1e-12 {
		t.Fatalf("R2 = %g, want %g", got, want)
	}
}

func TestRegressionR2Perfect(t *testing.T) {
	var r Regression
	values := []float32{1, 2, 3, 4}
	if err := r.Update(values, values); err != nil {
		t.Fatal(err)
	}
	if got := r.R2(); got != 1 {
		t.Fatalf("R2 = %g, want 1", got)
	}
}

func TestRegressionR2ConstantTarget(t *testing.T) {
	var r Regression
	if err := r.Update([]float32{1, 2, 3}, []float32{2, 2, 2}); err != nil {
		t.Fatal(err)
	}
	if got := r.R2(); got != 0 {
		t.Fatalf("R2 = %g, want 0 for a constant target", got)
	}
}

func TestRegressionR2Empty(t *testing.T) {
	var r Regression
	if got := r.R2(); got != 0 {
		t.Fatalf("R2 = %g, want 0", got)
	}
}

func TestRegressionR2AcrossBatches(t *testing.T) {
	var whole, split Regression
	prediction := []float32{1, 2, 3, 4, 5, 6}
	target := []float32{1, 3, 2, 5, 4, 7}
	if err := whole.Update(prediction, target); err != nil {
		t.Fatal(err)
	}
	if err := split.Update(prediction[:2], target[:2]); err != nil {
		t.Fatal(err)
	}
	if err := split.Update(prediction[2:], target[2:]); err != nil {
		t.Fatal(err)
	}
	if math.Abs(whole.R2()-split.R2()) > 1e-12 {
		t.Fatalf("batched R2 = %g, whole R2 = %g", split.R2(), whole.R2())
	}
}
