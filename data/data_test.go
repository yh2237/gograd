package data

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"slices"
	"sync"
	"testing"
)

func TestSamplerMidEpochResume(t *testing.T) {
	options := SamplerOptions{Shuffle: true, Seed: 37}
	a, err := NewIndexSampler(19, options)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Advance(7); err != nil {
		t.Fatal(err)
	}
	payload, err := a.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewIndexSampler(19, options)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.UnmarshalBinary(payload); err != nil {
		t.Fatal(err)
	}
	for epoch := 0; epoch < 3; epoch++ {
		for {
			left, e1 := a.Peek(5)
			right, e2 := b.Peek(5)
			if !slices.Equal(left, right) || !errors.Is(e1, e2) {
				t.Fatal("resume changed remaining or future order")
			}
			if e1 == io.EOF {
				break
			}
			if e1 != nil {
				t.Fatal(e1)
			}
			if err := a.Advance(len(left)); err != nil {
				t.Fatal(err)
			}
			if err := b.Advance(len(right)); err != nil {
				t.Fatal(err)
			}
		}
		if err := a.NextEpoch(); err != nil {
			t.Fatal(err)
		}
		if err := b.NextEpoch(); err != nil {
			t.Fatal(err)
		}
	}
	state := a.State()
	state.Order[0] = -1
	if a.State().Order[0] < 0 {
		t.Fatal("state aliases sampler order")
	}
	before := a.State()
	for _, mutate := range []func(*IndexSamplerState){
		func(s *IndexSamplerState) { s.Order[0] = s.Order[1] },
		func(s *IndexSamplerState) { s.Position = 20 },
		func(s *IndexSamplerState) { s.Seed++ },
		func(s *IndexSamplerState) { s.Random = []byte{1} },
		func(s *IndexSamplerState) { s.Version++ },
	} {
		invalid := a.State()
		mutate(&invalid)
		if err := a.LoadState(invalid); err == nil {
			t.Fatal("invalid sampler state accepted")
		}
		if !reflect.DeepEqual(before, a.State()) {
			t.Fatal("failed restore mutated state")
		}
	}
}

type failureDataset struct {
	values SliceDataset[int]
	get    func(context.Context, int) error
}

func (d failureDataset) Len() int { return len(d.values) }
func (d failureDataset) Get(ctx context.Context, i int) (int, error) {
	if d.get != nil {
		if err := d.get(ctx, i); err != nil {
			return 0, err
		}
	}
	return d.values.Get(ctx, i)
}

func TestLoaderRetryCancelAndCheckpoint(t *testing.T) {
	ctx := context.Background()
	failure := errors.New("read failed")
	sampler, _ := NewIndexSampler(5, SamplerOptions{})
	failRead, failCollate := true, false
	dataset := failureDataset{values: SliceDataset[int]{10, 11, 12, 13, 14}, get: func(_ context.Context, i int) error {
		if failRead && i == 1 {
			return failure
		}
		return nil
	}}
	collate := func(v []int) ([]int, error) {
		if failCollate {
			return nil, io.EOF
		}
		return slices.Clone(v), nil
	}
	loader, err := NewLoader(dataset, sampler, collate, LoaderOptions{BatchSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	before, _ := loader.State()
	if _, err := loader.Next(ctx); !errors.Is(err, failure) {
		t.Fatal("read error lost")
	}
	after, _ := loader.State()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("read failure consumed a batch")
	}
	failRead = false
	failCollate = true
	if _, err := loader.Next(ctx); !errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		t.Fatal("callback EOF was mistaken for epoch end")
	}
	after, _ = loader.State()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("collate failure consumed a batch")
	}
	failCollate = false
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := loader.Next(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation lost")
	}
	batch, err := loader.Next(ctx)
	if err != nil || !slices.Equal(batch, []int{10, 11}) {
		t.Fatal("retry skipped input")
	}
	state, err := loader.State()
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(state)
	var restoredState LoaderState
	if err := json.Unmarshal(encoded, &restoredState); err != nil {
		t.Fatal(err)
	}
	restoredSampler, _ := NewIndexSampler(5, SamplerOptions{})
	restored, _ := NewLoader(dataset, restoredSampler, collate, LoaderOptions{BatchSize: 2})
	if err := restored.LoadState(restoredState); err != nil {
		t.Fatal(err)
	}
	for _, want := range [][]int{{12, 13}, {14}} {
		got, err := restored.Next(ctx)
		if err != nil || !slices.Equal(got, want) {
			t.Fatalf("restored %v %v", got, err)
		}
	}
	if _, err := restored.Next(ctx); err != io.EOF {
		t.Fatal("epoch did not terminate")
	}
	bad := restoredState
	bad.BatchSize++
	before, _ = restored.State()
	if err := restored.LoadState(bad); err == nil {
		t.Fatal("incompatible loader state accepted")
	}
	after, _ = restored.State()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("failed loader restore mutated state")
	}
	// Cancellation during collation must leave the delivered-batch cursor intact.
	active, stop := context.WithCancel(ctx)
	l, _ := NewLoader(dataset, restoredSampler, func(v []int) ([]int, error) { stop(); return v, nil }, LoaderOptions{BatchSize: 2})
	if err := l.NextEpoch(); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Next(active); !errors.Is(err, context.Canceled) || restoredSampler.State().Position != 0 {
		t.Fatal("cancelled collate advanced cursor")
	}
}

func TestSplitSubsetDropLastAndEmpty(t *testing.T) {
	train, valid, err := RandomSplit(17, 5, 8)
	if err != nil {
		t.Fatal(err)
	}
	a, b, _ := RandomSplit(17, 5, 8)
	if !slices.Equal(a, train) || !slices.Equal(b, valid) {
		t.Fatal("split not deterministic")
	}
	all := append(slices.Clone(train), valid...)
	slices.Sort(all)
	for i, v := range all {
		if i != v {
			t.Fatal("split did not cover dataset exactly once")
		}
	}
	dataset := SliceDataset[int]{0, 1, 2, 3, 4}
	indices := []int{4, 2, 0}
	subset, err := NewSubset[int](dataset, indices)
	if err != nil {
		t.Fatal(err)
	}
	indices[0] = 1
	v, err := subset.Get(context.Background(), 0)
	if err != nil || v != 4 {
		t.Fatal("subset indices were borrowed")
	}
	sampler, _ := NewIndexSampler(subset.Len(), SamplerOptions{})
	loader, _ := NewLoader(subset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2, DropLast: true})
	batch, err := loader.Next(context.Background())
	if err != nil || !slices.Equal(batch, []int{4, 2}) {
		t.Fatal("subset mapping changed")
	}
	if _, err := loader.Next(context.Background()); err != io.EOF || sampler.State().Position != 3 {
		t.Fatal("drop-last tail was not consumed")
	}
	empty, _ := NewIndexSampler(0, SamplerOptions{})
	l, _ := NewLoader(SliceDataset[int]{}, empty, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2})
	if _, err := l.Next(context.Background()); err != io.EOF {
		t.Fatal("empty dataset did not finish")
	}
	if _, err := NewLoader(dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2}); err == nil {
		t.Fatal("size mismatch accepted")
	}
	if _, _, err := RandomSplit(4, 5, 0); err == nil {
		t.Fatal("invalid split accepted")
	}
}

func TestFloatCollates(t *testing.T) {
	samples := []FloatSample{{[]float32{1, 2, 3, 4}, []int{2, 2}, 7}, {[]float32{5, 6}, []int{1, 2}, 8}, {nil, []int{0, 2}, 9}}
	batch, err := PadSequences(samples, PaddingOptions{Value: -1})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(batch.Shape, []int{3, 2, 2}) || !slices.Equal(batch.Values, []float32{1, 2, 3, 4, 5, 6, -1, -1, -1, -1, -1, -1}) || !slices.Equal(batch.Lengths, []int{2, 1, 0}) || !slices.Equal(batch.Mask, []bool{true, true, true, false, false, false}) {
		t.Fatal("padding values or mask incorrect")
	}
	truncated, err := PadSequences(samples, PaddingOptions{Length: 1, Truncate: true})
	if err != nil || !slices.Equal(truncated.Lengths, []int{1, 1, 0}) {
		t.Fatal("explicit truncation failed")
	}
	if _, err := PadSequences(samples, PaddingOptions{Length: 1}); err == nil {
		t.Fatal("implicit truncation accepted")
	}
	empty, err := PadSequences(samples[2:], PaddingOptions{})
	if err != nil || len(empty.Values) != 0 || !slices.Equal(empty.Shape, []int{1, 0, 2}) {
		t.Fatal("empty sequence handling failed")
	}
	stacked, err := Stack([]FloatSample{samples[0], samples[0]})
	if err != nil {
		t.Fatal(err)
	}
	stacked.Values[0] = 99
	if samples[0].Values[0] != 1 {
		t.Fatal("collate borrowed sample storage")
	}
	if _, err := Stack(samples); err == nil {
		t.Fatal("mismatched stack shape accepted")
	}
	if _, err := Stack([]FloatSample{{nil, []int{-1}, 0}}); err == nil {
		t.Fatal("invalid shape accepted")
	}
}

func TestIndependentLoaders(t *testing.T) {
	dataset := SliceDataset[int]{0, 1, 2, 3, 4, 5, 6, 7, 8}
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sampler, _ := NewIndexSampler(len(dataset), SamplerOptions{Shuffle: true, Seed: 3})
			loader, _ := NewLoader(dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 4})
			for epoch := 0; epoch < 5; epoch++ {
				var seen []int
				for {
					batch, err := loader.Next(context.Background())
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Error(err)
						return
					}
					seen = append(seen, batch...)
				}
				slices.Sort(seen)
				if !slices.Equal(seen, []int(dataset)) {
					t.Error("independent epoch lost input")
				}
				if err := loader.NextEpoch(); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
