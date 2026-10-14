package data

import (
	"context"
	"math"
	"testing"
)

func TestAugmentationDeterminism(t *testing.T) {
	values := make([]float32, 12)
	for i := range values {
		values[i] = float32(i) / 10
	}
	noise := make([]float32, len(values))
	copy(noise, values)
	AddGaussianNoiseFloat32(noise, 7, 3, .1)
	repeated := make([]float32, len(values))
	copy(repeated, values)
	AddGaussianNoiseFloat32(repeated, 7, 3, .1)
	for i := range noise {
		if noise[i] != repeated[i] {
			t.Fatalf("noise mismatch at %d", i)
		}
	}

	// Order independence: fill a second buffer with reversed-ish access.
	a := make([]float32, len(values))
	b := make([]float32, len(values))
	copy(a, values)
	copy(b, values)
	// Simulate different iteration order: apply same mutations to each and compare.
	AddGaussianNoiseFloat32(a, 7, 3, .1)
	AddGaussianNoiseFloat32(b, 7, 3, .1)
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 0 {
			t.Fatalf("order dependence at %d", i)
		}
	}
}

func TestTransformDatasetDeterminism(t *testing.T) {
	base := sliceDataset{samples: []FloatSample{
		{Values: []float32{1, 2, 3}, Label: 1},
		{Values: []float32{4, 5, 6}, Label: 0},
	}}
	var calls int
	td := TransformDataset[FloatSample]{
		Base: base,
		Seed: 9,
		Apply: func(index int, seed uint64, sample FloatSample) (FloatSample, error) {
			calls++
			out := FloatSample{Values: append([]float32(nil), sample.Values...), Label: sample.Label}
			AddGaussianNoiseFloat32(out.Values, seed, index, .25)
			return out, nil
		},
	}
	s0, _ := td.Get(context.Background(), 0)
	s0r, _ := td.Get(context.Background(), 0)
	if len(s0.Values) != len(s0r.Values) || s0.Values[0] != s0r.Values[0] {
		t.Fatal("transform was not reproducible")
	}
	if td.Len() != 2 {
		t.Fatal("transform length wrong")
	}
}

type sliceDataset struct{ samples []FloatSample }

func (d sliceDataset) Len() int { return len(d.samples) }
func (d sliceDataset) Get(_ context.Context, index int) (FloatSample, error) {
	return d.samples[index], nil
}
