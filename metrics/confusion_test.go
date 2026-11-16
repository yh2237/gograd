package metrics

import (
	"math"
	"testing"
)

func TestConfusionMatrixCounts(t *testing.T) {
	var c ConfusionMatrix
	// actual:    0 0 1 1 2
	// predicted: 0 1 1 2 2
	if err := c.Update([]int{0, 1, 1, 2, 2}, []int{0, 0, 1, 1, 2}, 3, -1); err != nil {
		t.Fatal(err)
	}
	if c.Count() != 5 || c.Classes() != 3 {
		t.Fatalf("count=%d classes=%d, want 5/3", c.Count(), c.Classes())
	}
	want := [][]int{{1, 1, 0}, {0, 1, 1}, {0, 0, 1}}
	got := c.Matrix()
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("matrix[%d][%d] = %d, want %d", i, j, got[i][j], want[i][j])
			}
		}
	}
	for class, wantSupport := range []int{2, 2, 1} {
		if got := c.Support(class); got != wantSupport {
			t.Fatalf("support[%d] = %d, want %d", class, got, wantSupport)
		}
	}
}

func TestConfusionMatrixMetrics(t *testing.T) {
	var c ConfusionMatrix
	// class 0: 2 actual, 1 predicted correctly, 1 predicted as 1
	// class 1: 2 actual, 1 predicted correctly, 1 predicted as 2
	// class 2: 1 actual, 1 predicted correctly
	if err := c.Update([]int{0, 1, 1, 2, 2}, []int{0, 0, 1, 1, 2}, 3, -1); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		class             int
		precision, recall float64
		f1                float64
	}{
		{0, 1, 0.5, 2.0 / 3},
		{1, 0.5, 0.5, 0.5},
		{2, 0.5, 1, 2.0 / 3},
	} {
		if got := c.Precision(tc.class); math.Abs(got-tc.precision) > 1e-12 {
			t.Fatalf("precision[%d] = %g, want %g", tc.class, got, tc.precision)
		}
		if got := c.Recall(tc.class); math.Abs(got-tc.recall) > 1e-12 {
			t.Fatalf("recall[%d] = %g, want %g", tc.class, got, tc.recall)
		}
		if got := c.F1(tc.class); math.Abs(got-tc.f1) > 1e-12 {
			t.Fatalf("f1[%d] = %g, want %g", tc.class, got, tc.f1)
		}
	}
	wantMacro := (2.0/3 + 0.5 + 2.0/3) / 3
	if got := c.MacroF1(); math.Abs(got-wantMacro) > 1e-12 {
		t.Fatalf("macroF1 = %g, want %g", got, wantMacro)
	}
}

func TestConfusionMatrixIgnoresIndex(t *testing.T) {
	var c ConfusionMatrix
	if err := c.Update([]int{0, 2, 1}, []int{0, 2, 1}, 3, 2); err != nil {
		t.Fatal(err)
	}
	if c.Count() != 2 {
		t.Fatalf("count = %d, want 2", c.Count())
	}
	if got := c.Support(2); got != 0 {
		t.Fatalf("support[2] = %d, want 0", got)
	}
	if got := c.MacroF1(); got != 1 {
		t.Fatalf("macroF1 = %g, want 1", got)
	}
}

func TestConfusionMatrixEmpty(t *testing.T) {
	var c ConfusionMatrix
	if c.Count() != 0 || c.Classes() != 0 || c.Matrix() != nil {
		t.Fatal("zero accumulator should be empty")
	}
	if c.Precision(0) != 0 || c.Recall(0) != 0 || c.F1(0) != 0 || c.MacroF1() != 0 {
		t.Fatal("zero accumulator should report zero metrics")
	}
}

func TestConfusionMatrixRejectsInvalid(t *testing.T) {
	var c ConfusionMatrix
	for _, tc := range []struct {
		name              string
		predicted, actual []int
		classes, ignore   int
	}{
		{"zero classes", []int{0}, []int{0}, 0, -1},
		{"length mismatch", []int{0}, []int{0, 1}, 2, -1},
		{"label out of range", []int{0}, []int{5}, 2, -1},
		{"prediction out of range", []int{5}, []int{0}, 2, -1},
		{"negative label", []int{0}, []int{-2}, 2, -1},
		{"negative prediction", []int{-2}, []int{0}, 2, -1},
	} {
		if err := c.Update(tc.predicted, tc.actual, tc.classes, tc.ignore); err == nil {
			t.Fatalf("%s: expected an error", tc.name)
		}
		if c.Count() != 0 || c.Classes() != 0 {
			t.Fatalf("%s: accumulator changed after an error", tc.name)
		}
	}
}

func TestConfusionMatrixRejectsClassChange(t *testing.T) {
	var c ConfusionMatrix
	if err := c.Update([]int{0}, []int{0}, 2, -1); err != nil {
		t.Fatal(err)
	}
	if err := c.Update([]int{0}, []int{0}, 3, -1); err == nil {
		t.Fatal("expected an error when the class count changes")
	}
	if c.Classes() != 2 || c.Count() != 1 {
		t.Fatalf("classes=%d count=%d, want 2/1", c.Classes(), c.Count())
	}
}

func TestConfusionMatrixReset(t *testing.T) {
	var c ConfusionMatrix
	if err := c.Update([]int{0, 1}, []int{0, 1}, 2, -1); err != nil {
		t.Fatal(err)
	}
	c.Reset()
	if c.Count() != 0 || c.Classes() != 0 || c.Matrix() != nil {
		t.Fatal("reset should clear the accumulator")
	}
	if err := c.Update([]int{0}, []int{0}, 3, -1); err != nil {
		t.Fatalf("reset should allow a new class count: %v", err)
	}
}
