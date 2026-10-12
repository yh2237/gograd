package data

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func TestParseNumericCSV(t *testing.T) {
	table, err := ParseNumericCSV(strings.NewReader("# comment\nx,y,z\n1,2,3\n4,5,6\n"), ParseOptions{Comment: '#'})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(table.Names, []string{"x", "y", "z"}) || table.Len() != 2 || table.Columns() != 3 {
		t.Fatalf("table %+v", table)
	}
	if table.Rows[1][2] != 6 {
		t.Fatalf("row %v", table.Rows[1])
	}
	noHeader, err := ParseNumericCSV(strings.NewReader("1\t2\n3\t4\n"), ParseOptions{Delimiter: '\t', NoHeader: true})
	if err != nil {
		t.Fatal(err)
	}
	if noHeader.Len() != 2 || len(noHeader.Names) != 2 || noHeader.Names[0] != "" {
		t.Fatalf("no-header table %+v", noHeader)
	}
}

func TestParseNumericCSVErrors(t *testing.T) {
	for name, source := range map[string]string{
		"ragged":     "a,b\n1,2\n3,4,5\n",
		"nonnumeric": "a,b\n1,two\n",
		"nonfinite":  "a,b\n1,NaN\n",
		"empty":      "",
		"justblank":  "\n\n",
	} {
		if _, err := ParseNumericCSV(strings.NewReader(source), ParseOptions{}); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if _, err := ParseNumericCSV(nil, ParseOptions{}); err == nil {
		t.Fatal("nil reader accepted")
	}
}

func TestRegressionSamplesAndStack(t *testing.T) {
	table, err := ParseNumericCSV(strings.NewReader("f1,f2,f3,target\n1,2,3,10\n4,5,6,20\n"), ParseOptions{})
	if err != nil {
		t.Fatal(err)
	}
	samples, err := table.RegressionSamples(nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 || samples[0].Target != 10 || len(samples[0].Features) != 3 || samples[1].Features[2] != 6 {
		t.Fatalf("samples %+v", samples)
	}
	// Explicit columns reorder features.
	reordered, err := table.RegressionSamples([]int{2, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if reordered[0].Features[0] != 3 || reordered[0].Features[1] != 1 || reordered[0].Target != 2 {
		t.Fatalf("reordered %+v", reordered[0])
	}
	for name, test := range map[string]struct {
		features []int
		target   int
	}{
		"target is feature": {[]int{0, 3}, 3},
		"duplicate":         {[]int{0, 0}, 3},
		"out of range":      {[]int{9}, 3},
		"bad target":        {nil, 7},
	} {
		if _, err := table.RegressionSamples(test.features, test.target); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	batch, err := StackRegression(samples)
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Values) != 6 || batch.Shape[0] != 2 || batch.Shape[1] != 3 || batch.Targets[1] != 20 {
		t.Fatalf("batch %+v", batch)
	}
	if _, err := StackRegression(nil); err == nil {
		t.Fatal("empty stack accepted")
	}
	if _, err := StackRegression([]RegressionSample{{Features: []float32{1}}, {Features: []float32{1, 2}}}); err == nil {
		t.Fatal("unequal features accepted")
	}
}

func TestStandardizerFitApplyState(t *testing.T) {
	samples := []RegressionSample{
		{Features: []float32{1, 5}, Target: 0},
		{Features: []float32{3, 5}, Target: 0},
	}
	standardizer, err := FitStandardizer(samples)
	if err != nil {
		t.Fatal(err)
	}
	if standardizer.Mean[0] != 2 || standardizer.Std[0] != 1 {
		t.Fatalf("statistics %+v", standardizer)
	}
	// A constant feature gets a unit scale and becomes zero.
	if standardizer.Std[1] != 1 {
		t.Fatalf("constant feature scale %v", standardizer.Std[1])
	}
	applied := standardizer.Apply(samples[0])
	if applied.Features[0] != -1 || applied.Features[1] != 0 || applied.Target != samples[0].Target {
		t.Fatalf("applied %+v", applied)
	}
	batch, err := StackRegression(samples)
	if err != nil {
		t.Fatal(err)
	}
	if err := standardizer.ApplyBatch(&batch); err != nil {
		t.Fatal(err)
	}
	if batch.Values[0] != -1 || batch.Values[2] != 1 {
		t.Fatalf("batch values %v", batch.Values)
	}
	if err := (Standardizer{}).ApplyBatch(&batch); err == nil {
		t.Fatal("mismatched batch accepted")
	}
	state := standardizer.State()
	var restored Standardizer
	if err := restored.LoadState(state); err != nil {
		t.Fatal(err)
	}
	if restored.Mean[0] != standardizer.Mean[0] || restored.Std[0] != standardizer.Std[0] {
		t.Fatalf("restored %+v", restored)
	}
	for name, bad := range map[string]StandardizerState{
		"version":   {Version: 2, Mean: []float32{0}, Std: []float32{1}},
		"size":      {Version: 1, Mean: []float32{0}, Std: nil},
		"zero":      {Version: 1, Mean: []float32{0}, Std: []float32{0}},
		"nonfinite": {Version: 1, Mean: []float32{float32(math.Inf(1))}, Std: []float32{1}},
	} {
		if err := restored.LoadState(bad); err == nil {
			t.Fatalf("%s state accepted", name)
		}
	}
	if _, err := FitStandardizer(nil); err == nil {
		t.Fatal("empty fit accepted")
	}
	if _, err := FitStandardizer([]RegressionSample{{Features: []float32{1}}, {Features: []float32{1, 2}}}); err == nil {
		t.Fatal("inconsistent fit accepted")
	}
}
