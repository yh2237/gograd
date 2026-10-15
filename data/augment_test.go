package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	randv2 "math/rand/v2"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"
)

func TestHorizontalFlipPlanes(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	want := []float32{3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10}
	original := slices.Clone(values)
	if err := FlipsHorizontallyFloat32(values, 2, 2, 3); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, want) {
		t.Fatalf("flipped %v, want %v", values, want)
	}
	if err := FlipsHorizontallyFloat32(values, 2, 2, 3); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, original) {
		t.Fatal("double flip changed image")
	}
	if err := FlipsHorizontallyFloat32(values, 2, 6, 1); err != nil || !slices.Equal(values, original) {
		t.Fatal("width-one flip changed image", err)
	}
	if err := FlipsHorizontallyFloat32(values, 1, 3, 4); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, []float32{4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9}) {
		t.Fatal("even-width flip incorrect")
	}
}

func TestRandomFlipUsesFullSeed(t *testing.T) {
	original := []float32{1, 2, 3, 4}
	seed := uint64(1)<<63 | 19
	for _, probability := range []float64{0, .5, 1} {
		for index := 0; index < 20; index++ {
			values := slices.Clone(original)
			want := slices.Clone(original)
			rng := randv2.New(randv2.NewPCG(seed, uint64(index)^0x9e3779b97f4a7c15))
			if rng.Float64() < probability {
				want = []float32{2, 1, 4, 3}
			}
			if err := RandomHorizontalFlipFloat32(values, 1, 2, 2, seed, index, probability); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, want) {
				t.Fatalf("seed=%d index=%d p=%g: %v != %v", seed, index, probability, values, want)
			}
		}
	}
	legacy, current := slices.Clone(original), slices.Clone(original)
	if err := FlipEarly(legacy, 1, 2, 2, 7, 3, .5); err != nil {
		t.Fatal(err)
	}
	if err := RandomHorizontalFlipFloat32(current, 1, 2, 2, 7, 3, .5); err != nil || !slices.Equal(legacy, current) {
		t.Fatal("legacy seed mapping changed", err)
	}
}

func TestAugmentationRejectsInvalidInputsWithoutMutation(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for name, transform := range map[string]func([]float32) error{
		"channels":             func(v []float32) error { return FlipsHorizontallyFloat32(v, 0, 2, 2) },
		"negative":             func(v []float32) error { return FlipsHorizontallyFloat32(v, 1, -2, 2) },
		"short":                func(v []float32) error { return FlipsHorizontallyFloat32(v, 2, 2, 2) },
		"long":                 func(v []float32) error { return FlipsHorizontallyFloat32(v, 1, 1, 2) },
		"overflow":             func(v []float32) error { return FlipsHorizontallyFloat32(v, maxInt, 2, 2) },
		"shape-even-at-p-zero": func(v []float32) error { return RandomHorizontalFlipFloat32(v, 1, 0, 2, 7, 0, 0) },
		"flip-index":           func(v []float32) error { return RandomHorizontalFlipFloat32(v, 1, 2, 2, 7, -1, .5) },
		"prob-negative":        func(v []float32) error { return RandomHorizontalFlipFloat32(v, 1, 2, 2, 7, 0, -.1) },
		"prob-large":           func(v []float32) error { return RandomHorizontalFlipFloat32(v, 1, 2, 2, 7, 0, 1.1) },
		"prob-nan":             func(v []float32) error { return RandomHorizontalFlipFloat32(v, 1, 2, 2, 7, 0, math.NaN()) },
		"prob-inf":             func(v []float32) error { return RandomHorizontalFlipFloat32(v, 1, 2, 2, 7, 0, math.Inf(1)) },
		"noise-index":          func(v []float32) error { return AddGaussianNoiseFloat32(v, 7, -1, .1) },
		"noise-negative":       func(v []float32) error { return AddGaussianNoiseFloat32(v, 7, 0, -.1) },
		"noise-nan":            func(v []float32) error { return AddGaussianNoiseFloat32(v, 7, 0, float32(math.NaN())) },
		"noise-inf":            func(v []float32) error { return AddGaussianNoiseFloat32(v, 7, 0, float32(math.Inf(1))) },
	} {
		t.Run(name, func(t *testing.T) {
			values := []float32{1, 2, 3, 4}
			want := slices.Clone(values)
			if err := transform(values); err == nil {
				t.Fatal("invalid input accepted")
			}
			if !slices.Equal(values, want) {
				t.Fatal("error mutated values")
			}
		})
	}
	for _, invalid := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, std := range []float32{0, .1} {
			values := []float32{1, invalid}
			if err := AddGaussianNoiseFloat32(values, 7, 0, std); err == nil || values[0] != 1 || math.Float32bits(values[1]) != math.Float32bits(invalid) {
				t.Fatal("nonfinite input accepted or partially mutated", err)
			}
		}
	}
	values := make([]float32, 64)
	for i := range values {
		values[i] = math.MaxFloat32
	}
	want := slices.Clone(values)
	if err := AddGaussianNoiseFloat32(values, 0, 0, math.MaxFloat32); err == nil || !slices.Equal(values, want) {
		t.Fatal("overflow accepted or partially mutated", err)
	}
}

func TestGaussianNoiseMatchesAudioFormula(t *testing.T) {
	original := []float32{-.25, 0, .125, .5}
	for _, seed := range []uint64{0, 7, math.MaxUint64} {
		for _, index := range []int{0, 3, 99} {
			for _, std := range []float32{0, .15} {
				want, values := slices.Clone(original), slices.Clone(original)
				rng := randv2.New(randv2.NewPCG(seed, uint64(index)+0x9e3779b97f4a7c15))
				for i := range want {
					want[i] += float32(rng.NormFloat64()) * std
				}
				if err := AddGaussianNoiseFloat32(values, seed, index, std); err != nil || !slices.Equal(values, want) {
					t.Fatalf("audio formula changed seed=%d index=%d std=%g: %v", seed, index, std, err)
				}
			}
		}
	}
	if err := AddGaussianNoiseFloat32(nil, 7, 0, .15); err != nil {
		t.Fatal("empty signal rejected", err)
	}
}

func TestTransformDatasetValidationAndErrors(t *testing.T) {
	ctx := context.Background()
	base := SliceDataset[int]{1, 2}
	calls := 0
	dataset := TransformDataset[int]{Base: base, Seed: math.MaxUint64, Apply: func(index int, seed uint64, sample int) (int, error) {
		calls++
		if index != 1 || seed != math.MaxUint64 || sample != 2 {
			t.Fatal("callback did not receive unchanged index/seed/sample")
		}
		return sample + 1, nil
	}}
	if got, err := dataset.Get(ctx, 1); err != nil || got != 3 || calls != 1 {
		t.Fatal("transform failed", got, err, calls)
	}
	for _, index := range []int{-1, 2} {
		if _, err := dataset.Get(ctx, index); err == nil {
			t.Fatal("invalid index accepted")
		}
	}
	if _, err := dataset.Get(nil, 1); err == nil || calls != 1 {
		t.Fatal("invalid input reached Apply")
	}
	var empty TransformDataset[int]
	if empty.Len() != 0 {
		t.Fatal("nil dataset length")
	}
	if _, err := empty.Get(ctx, 0); err == nil {
		t.Fatal("nil dataset accepted")
	}
	dataset.Apply = nil
	if got, err := dataset.Get(ctx, 0); err != nil || got != 1 {
		t.Fatal("nil transform did not pass through", err)
	}
	failure := errors.New("transform failure")
	dataset.Apply = func(int, uint64, int) (int, error) { return 9, failure }
	if got, err := dataset.Get(ctx, 0); !errors.Is(err, failure) || got != 0 {
		t.Fatal("callback error did not return zero/error", got, err)
	}
	broken := TransformDataset[int]{Base: failureDataset{values: base, get: func(context.Context, int) error { return failure }}}
	if _, err := broken.Get(ctx, 0); !errors.Is(err, failure) {
		t.Fatal("base error lost", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	dataset.Apply = func(int, uint64, int) (int, error) { cancel(); return 9, nil }
	if got, err := dataset.Get(canceled, 0); !errors.Is(err, context.Canceled) || got != 0 {
		t.Fatal("cancellation during transform lost", err)
	}
	if _, err := dataset.Get(canceled, 0); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled context accepted", err)
	}
}

func TestAugmentedLoaderWorkersResumeAndRetry(t *testing.T) {
	base := make(SliceDataset[FloatSample], 13)
	for i := range base {
		base[i] = FloatSample{Values: []float32{1, 2, 3, 4, 5, 6}, Shape: []int{1, 2, 3}, Label: i}
	}
	original := make(SliceDataset[FloatSample], len(base))
	for i, sample := range base {
		original[i] = FloatSample{Values: slices.Clone(sample.Values), Shape: slices.Clone(sample.Shape), Label: sample.Label}
	}
	dataset := TransformDataset[FloatSample]{Base: base, Seed: uint64(1)<<63 | 7, Apply: func(index int, seed uint64, sample FloatSample) (FloatSample, error) {
		out := FloatSample{Values: slices.Clone(sample.Values), Shape: slices.Clone(sample.Shape), Label: sample.Label}
		if err := RandomHorizontalFlipFloat32(out.Values, 1, 2, 3, seed, index, .5); err != nil {
			return FloatSample{}, err
		}
		return out, AddGaussianNoiseFloat32(out.Values, seed, index, .15)
	}}
	newLoader := func(t *testing.T, workers int) *Loader[FloatSample, FloatBatch] {
		t.Helper()
		sampler, err := NewIndexSampler(len(base), SamplerOptions{Shuffle: true, Seed: 19})
		if err != nil {
			t.Fatal(err)
		}
		loader, err := NewLoader[FloatSample, FloatBatch](dataset, sampler, Stack, LoaderOptions{BatchSize: 3, Workers: workers})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { loader.Close() })
		return loader
	}
	ctx := context.Background()
	for _, workers := range []int{0, 2, 4} {
		t.Run(fmt.Sprint(workers), func(t *testing.T) {
			baseline, interrupted := newLoader(t, 0), newLoader(t, workers)
			for i := 0; i < 2; i++ {
				want, err := baseline.Next(ctx)
				if err != nil {
					t.Fatal(err)
				}
				got, err := interrupted.Next(ctx)
				if err != nil || !reflect.DeepEqual(want, got) {
					t.Fatal("worker count changed augmented batch", err)
				}
			}
			canceled, cancel := context.WithCancel(ctx)
			cancel()
			if _, err := interrupted.Next(canceled); !errors.Is(err, context.Canceled) {
				t.Fatal("cancel not propagated", err)
			}
			state, err := interrupted.State()
			if err != nil {
				t.Fatal(err)
			}
			restored := newLoader(t, (workers+2)%5)
			if err := restored.LoadState(state); err != nil {
				t.Fatal(err)
			}
			for epoch := 0; epoch < 2; epoch++ {
				for {
					want, e1 := baseline.Next(ctx)
					got, e2 := restored.Next(ctx)
					if !errors.Is(e1, e2) || !reflect.DeepEqual(want, got) {
						t.Fatal("resume changed augmentation/order", e1, e2)
					}
					if e1 == io.EOF {
						break
					}
					if e1 != nil {
						t.Fatal(e1)
					}
				}
				if err := baseline.NextEpoch(); err != nil {
					t.Fatal(err)
				}
				if err := restored.NextEpoch(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	if !reflect.DeepEqual(base, original) {
		t.Fatal("augmentation mutated borrowed dataset memory")
	}
}

func TestTransformFailureLeavesLoaderPosition(t *testing.T) {
	for _, workers := range []int{0, 3} {
		t.Run(fmt.Sprint(workers), func(t *testing.T) {
			failure := errors.New("augmentation failed")
			var fail atomic.Bool
			fail.Store(true)
			dataset := TransformDataset[int]{Base: SliceDataset[int]{0, 1, 2, 3}, Apply: func(index int, seed uint64, sample int) (int, error) {
				if index == 0 && fail.Load() {
					return 0, failure
				}
				return sample, nil
			}}
			sampler, err := NewIndexSampler(dataset.Len(), SamplerOptions{})
			if err != nil {
				t.Fatal(err)
			}
			loader, err := NewLoader[int, []int](dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2, Workers: workers})
			if err != nil {
				t.Fatal(err)
			}
			defer loader.Close()
			if _, err := loader.Next(context.Background()); !errors.Is(err, failure) || sampler.State().Position != 0 {
				t.Fatal("failure advanced data position", err)
			}
			fail.Store(false)
			if got, err := loader.Next(context.Background()); err != nil || !slices.Equal(got, []int{0, 1}) {
				t.Fatal("retry skipped failed samples", got, err)
			}
		})
	}
}

func TestTransformBeforeSubsetPreservesOriginalIndex(t *testing.T) {
	base := SliceDataset[int]{10, 20, 30, 40}
	dataset := TransformDataset[int]{Base: base, Seed: 9, Apply: func(index int, seed uint64, sample int) (int, error) {
		return sample + index*100 + int(seed), nil
	}}
	subset, err := NewSubset[int](dataset, []int{3, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	for _, index := range []int{2, 1, 0, 1, 2} {
		got, err := subset.Get(context.Background(), index)
		want := []int{349, 129, 349}[index]
		if err != nil || got != want {
			t.Fatal("subset changed augmentation identity", got, want, err)
		}
	}
}
