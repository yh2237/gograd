package metrics

import (
	"math"
	"testing"
)

func TestMeanUnequalBatches(t *testing.T) {
	var m Mean
	if m.Count() != 0 || m.Value() != 0 {
		t.Fatal("nonzero empty mean")
	}
	for _, batch := range []struct {
		value  float64
		weight int
	}{{2, 4}, {7, 1}, {9, 0}} {
		if err := m.Add(batch.value, batch.weight); err != nil {
			t.Fatal(err)
		}
	}
	if m.Count() != 5 || m.Value() != 3 {
		t.Fatalf("count=%d mean=%g, want 5/3", m.Count(), m.Value())
	}
	for _, bad := range []struct {
		value  float64
		weight int
	}{{0, -1}, {math.NaN(), 0}, {math.Inf(1), 1}, {math.MaxFloat64, 2}, {1, int(^uint(0) >> 1)}} {
		before := m
		if err := m.Add(bad.value, bad.weight); err == nil || m != before {
			t.Fatal("invalid mean input accepted or changed state")
		}
	}
	m.Reset()
	if m != (Mean{}) {
		t.Fatal("Reset failed")
	}
}

func TestAccuracyTiesIgnoreAndUnequalBatches(t *testing.T) {
	var a Accuracy
	if a.Value() != 0 || a.Count() != 0 {
		t.Fatal("nonzero empty accuracy")
	}
	if err := a.UpdateLogits([]float32{2, 2, 1, 0, 3, 1, 1, 0, 4}, []int{0, 0, -1}, 3, -1); err != nil {
		t.Fatal(err)
	}
	if err := a.UpdateLogits([]float32{-3, -2, -1}, []int{2}, 3, -1); err != nil {
		t.Fatal(err)
	}
	if a.Count() != 3 || a.Correct() != 2 || a.Value() != 2.0/3 {
		t.Fatalf("accuracy=%+v value=%g", a, a.Value())
	}
	for _, bad := range []struct {
		scores  []float32
		labels  []int
		classes int
	}{
		{nil, nil, 0},
		{[]float32{1, 2}, []int{0}, 3},
		{[]float32{1, 2}, nil, 2},
		{[]float32{1, 2, 3, 4}, []int{1, 2}, 2},
		{[]float32{1, 2}, []int{-2}, 2},
		{[]float32{float32(math.NaN()), 2}, []int{-1}, 2},
		{[]float32{1, float32(math.Inf(1))}, []int{0}, 2},
	} {
		before := a
		if err := a.UpdateLogits(bad.scores, bad.labels, bad.classes, -1); err == nil || a != before {
			t.Fatal("invalid classification input accepted or changed state")
		}
	}
	before := a
	if err := a.UpdateLogits(nil, nil, 2, -1); err != nil || a != before {
		t.Fatal("empty update changed accuracy")
	}
	a.Reset()
	if a != (Accuracy{}) {
		t.Fatal("Reset failed")
	}
}

func TestRegressionUnequalBatches(t *testing.T) {
	prediction := []float32{1, -2, 4, 6, 2}
	target := []float32{0, 0, 1, 2, 2}
	var whole, split Regression
	if split.MSE() != 0 || split.RMSE() != 0 || split.MAE() != 0 || split.Count() != 0 {
		t.Fatal("nonzero empty regression")
	}
	if err := whole.Update(prediction, target); err != nil {
		t.Fatal(err)
	}
	for _, span := range [][2]int{{0, 2}, {2, 4}, {4, 5}} {
		if err := split.Update(prediction[span[0]:span[1]], target[span[0]:span[1]]); err != nil {
			t.Fatal(err)
		}
	}
	if split != whole || split.Count() != 5 || split.MSE() != 6 || split.RMSE() != math.Sqrt(6) || split.MAE() != 2 {
		t.Fatalf("regression=%+v, whole=%+v", split, whole)
	}
	for _, bad := range []struct{ prediction, target []float32 }{
		{prediction, nil},
		{[]float32{1, float32(math.NaN())}, []float32{0, 0}},
		{[]float32{1, 2}, []float32{0, float32(math.Inf(-1))}},
	} {
		before := split
		if err := split.Update(bad.prediction, bad.target); err == nil || split != before {
			t.Fatal("invalid regression input accepted or changed state")
		}
	}
	before := split
	if err := split.Update(nil, nil); err != nil || split != before {
		t.Fatal("empty update changed regression")
	}
	split.Reset()
	if split != (Regression{}) {
		t.Fatal("Reset failed")
	}
}

func TestRegressionSubtractsInFloat64(t *testing.T) {
	var r Regression
	if err := r.Update([]float32{math.MaxFloat32}, []float32{-math.MaxFloat32}); err != nil {
		t.Fatal(err)
	}
	difference := float64(math.MaxFloat32) * 2
	if r.MSE() != difference*difference || r.MAE() != difference {
		t.Fatal("subtraction overflowed in float32")
	}
}

func TestMetricCountOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	a := Accuracy{count: maxInt}
	if err := a.UpdateLogits([]float32{1}, []int{0}, 1, -1); err == nil || a.count != maxInt {
		t.Fatal("accuracy count overflow accepted")
	}
	r := Regression{count: maxInt}
	if err := r.Update([]float32{1}, []float32{0}); err == nil || r.count != maxInt {
		t.Fatal("regression count overflow accepted")
	}
}
