package data

import (
	"context"
	"errors"
	"io"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func awaitSignal(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("reader synchronization timed out")
	}
}

func TestPrefetchOrderBoundAndConsumerCollation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	gate, ahead := make(chan struct{}), make(chan struct{})
	var aheadOnce sync.Once
	var reads, active, peak atomic.Int64
	dataset := failureDataset{values: make(SliceDataset[int], 12), get: func(ctx context.Context, index int) error {
		reads.Add(1)
		n := active.Add(1)
		defer active.Add(-1)
		for old := peak.Load(); n > old && !peak.CompareAndSwap(old, n); old = peak.Load() {
		}
		if index == 0 {
			select {
			case <-gate:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if index == 5 {
			aheadOnce.Do(func() { close(ahead) })
		}
		return nil
	}}
	for i := range dataset.values {
		dataset.values[i] = i
	}
	sampler, _ := NewIndexSampler(12, SamplerOptions{})
	var collated [][]int
	loader, err := NewLoader(dataset, sampler, func(values []int) ([]int, error) {
		collated = append(collated, slices.Clone(values))
		return values, nil
	}, LoaderOptions{BatchSize: 2, Workers: 2, Prefetch: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer loader.Close()
	type result struct {
		batch []int
		err   error
	}
	done := make(chan result, 1)
	go func() { batch, err := loader.Next(ctx); done <- result{batch, err} }()
	awaitSignal(t, ahead)
	if reads.Load() != 6 || peak.Load() != 2 {
		t.Fatalf("prefetch bound/concurrency reads=%d peak=%d", reads.Load(), peak.Load())
	}
	select {
	case <-done:
		t.Fatal("later result bypassed blocked first sample")
	default:
	}
	close(gate)
	first := <-done
	if first.err != nil || !slices.Equal(first.batch, []int{0, 1}) {
		t.Fatal("first batch order changed")
	}
	if len(collated) != 1 {
		t.Fatal("collation ran speculatively")
	}
	for _, want := range [][]int{{2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}} {
		batch, err := loader.Next(ctx)
		if err != nil || !slices.Equal(batch, want) {
			t.Fatalf("ordered batch %v %v", batch, err)
		}
	}
	if _, err := loader.Next(ctx); err != io.EOF {
		t.Fatal("prefetch epoch did not end")
	}
	if active.Load() != 0 {
		t.Fatal("EOF left active readers")
	}
}

func TestPrefetchCheckpointDiscardsOnlyUndelivered(t *testing.T) {
	options := SamplerOptions{Shuffle: true, Seed: 19}
	sampler, _ := NewIndexSampler(13, options)
	order := sampler.State().Order
	allowed := map[int]bool{order[0]: true, order[1]: true}
	var blocked atomic.Bool
	blocked.Store(true)
	var active atomic.Int64
	started := make(chan struct{})
	var once sync.Once
	dataset := failureDataset{values: make(SliceDataset[int], 13), get: func(ctx context.Context, index int) error {
		active.Add(1)
		defer active.Add(-1)
		if blocked.Load() && !allowed[index] {
			once.Do(func() { close(started) })
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	for i := range dataset.values {
		dataset.values[i] = i
	}
	collate := func(v []int) ([]int, error) { return slices.Clone(v), nil }
	loader, _ := NewLoader(dataset, sampler, collate, LoaderOptions{BatchSize: 2, Workers: 4, Prefetch: 3})
	defer loader.Close()
	first, err := loader.Next(context.Background())
	if err != nil || !slices.Equal(first, order[:2]) {
		t.Fatal("initial shuffled batch incorrect")
	}
	awaitSignal(t, started)
	state, err := loader.State()
	if err != nil {
		t.Fatal(err)
	}
	if active.Load() != 0 || sampler.State().Position != 2 {
		t.Fatal("checkpoint captured scheduled instead of delivered position")
	}
	blocked.Store(false)
	// A synchronous restore and a different worker count must preserve both
	// remaining current order and the following epoch's RNG choices.
	baselineSampler, _ := NewIndexSampler(13, options)
	baseline, _ := NewLoader(dataset, baselineSampler, collate, LoaderOptions{BatchSize: 2})
	defer baseline.Close()
	baseline.Next(context.Background())
	restoredSampler, _ := NewIndexSampler(13, options)
	restored, _ := NewLoader(dataset, restoredSampler, collate, LoaderOptions{BatchSize: 2, Workers: 2, Prefetch: 2})
	defer restored.Close()
	if err := restored.LoadState(state); err != nil {
		t.Fatal(err)
	}
	for epoch := 0; epoch < 2; epoch++ {
		for {
			want, e1 := baseline.Next(context.Background())
			got, e2 := restored.Next(context.Background())
			if !errors.Is(e1, e2) || !slices.Equal(want, got) {
				t.Fatal("prefetch restore changed sample order")
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
	left, _ := baseline.State()
	right, _ := restored.State()
	if !reflect.DeepEqual(left, right) {
		t.Fatal("worker settings affected logical checkpoint state")
	}
}

func TestPrefetchCancellationRetryAndClose(t *testing.T) {
	var blocked atomic.Bool
	blocked.Store(true)
	var active atomic.Int64
	started := make(chan struct{})
	var once sync.Once
	dataset := failureDataset{values: SliceDataset[int]{0, 1, 2, 3, 4}, get: func(ctx context.Context, index int) error {
		active.Add(1)
		defer active.Add(-1)
		if blocked.Load() {
			once.Do(func() { close(started) })
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	sampler, _ := NewIndexSampler(5, SamplerOptions{})
	loader, _ := NewLoader(dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2, Workers: 3, Prefetch: 2})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := loader.Next(ctx); done <- err }()
	awaitSignal(t, started)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal("cancel did not reach caller")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancel did not join readers")
	}
	if active.Load() != 0 || sampler.State().Position != 0 {
		t.Fatal("cancel leaked work or consumed samples")
	}
	blocked.Store(false)
	first, err := loader.Next(context.Background())
	if err != nil || !slices.Equal(first, []int{0, 1}) {
		t.Fatal("retry skipped cancelled input")
	}
	if err := loader.Close(); err != nil {
		t.Fatal(err)
	}
	loader.Close()
	if active.Load() != 0 || sampler.State().Position != 2 {
		t.Fatal("Close consumed lookahead or leaked readers")
	}
	if _, err := loader.Next(context.Background()); !errors.Is(err, ErrLoaderClosed) {
		t.Fatal("Next accepted a closed loader")
	}
}

func TestPrefetchOrderedErrorsAndRetry(t *testing.T) {
	firstErr, secondErr := errors.New("first sample"), errors.New("second sample")
	var fail atomic.Bool
	fail.Store(true)
	secondDone := make(chan struct{})
	var once sync.Once
	dataset := failureDataset{values: SliceDataset[int]{0, 1, 2, 3}, get: func(ctx context.Context, index int) error {
		if !fail.Load() {
			return nil
		}
		if index == 1 {
			once.Do(func() { close(secondDone) })
			return secondErr
		}
		if index == 0 {
			select {
			case <-secondDone:
			case <-ctx.Done():
				return ctx.Err()
			}
			return firstErr
		}
		return nil
	}}
	sampler, _ := NewIndexSampler(4, SamplerOptions{})
	loader, _ := NewLoader(dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2, Workers: 2})
	defer loader.Close()
	if _, err := loader.Next(context.Background()); !errors.Is(err, firstErr) || sampler.State().Position != 0 {
		t.Fatal("worker completion order changed error/position")
	}
	fail.Store(false)
	batch, err := loader.Next(context.Background())
	if err != nil || !slices.Equal(batch, []int{0, 1}) {
		t.Fatal("failed batch was not retried")
	}
	// Collate failure is also transactional and is only called by the consumer.
	failCollate := true
	loader.collate = func(v []int) ([]int, error) {
		if failCollate {
			return nil, io.EOF
		}
		return v, nil
	}
	if _, err := loader.Next(context.Background()); !errors.Is(err, io.ErrUnexpectedEOF) || sampler.State().Position != 2 {
		t.Fatal("failed collate consumed a prefetched batch")
	}
	failCollate = false
	batch, err = loader.Next(context.Background())
	if err != nil || !slices.Equal(batch, []int{2, 3}) {
		t.Fatal("collate retry changed batch")
	}
}

func TestPrefetchTailEmptyAndPanic(t *testing.T) {
	for _, size := range []int{0, 1, 5} {
		sampler, _ := NewIndexSampler(size, SamplerOptions{})
		loader, err := NewLoader(make(SliceDataset[int], size), sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2, DropLast: true, Workers: 3, Prefetch: 2})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for {
			_, err := loader.Next(context.Background())
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			count++
		}
		if count != size/2 || sampler.State().Position != size {
			t.Fatal("drop-last/empty prefetch consumed wrong count")
		}
		loader.Close()
	}
	dataset := failureDataset{values: SliceDataset[int]{1}, get: func(context.Context, int) error { panic("reader panic") }}
	sampler, _ := NewIndexSampler(1, SamplerOptions{})
	loader, _ := NewLoader(dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 1, Workers: 1})
	defer loader.Close()
	if _, err := loader.Next(context.Background()); err == nil || sampler.State().Position != 0 {
		t.Fatal("worker panic was not a transactional error")
	}
	if _, err := NewLoader(SliceDataset[int]{1}, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 1, Prefetch: 1}); err == nil {
		t.Fatal("synchronous prefetch configuration accepted")
	}
}

func TestPrefetchContextReplacementAndCollateCancel(t *testing.T) {
	dataset := SliceDataset[int]{0, 1, 2, 3, 4, 5}
	sampler, _ := NewIndexSampler(6, SamplerOptions{})
	loader, _ := NewLoader(dataset, sampler, func(v []int) ([]int, error) { return v, nil }, LoaderOptions{BatchSize: 2, Workers: 2, Prefetch: 2})
	defer loader.Close()
	ctx, cancel := context.WithCancel(context.Background())
	first, err := loader.Next(ctx)
	if err != nil || !slices.Equal(first, []int{0, 1}) {
		t.Fatal("first batch failed")
	}
	cancel()
	second, err := loader.Next(context.Background())
	if err != nil || !slices.Equal(second, []int{2, 3}) {
		t.Fatal("new context skipped old cancelled lookahead")
	}
	// Cancellation initiated by collation still cannot commit that batch.
	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	loader.collate = func(v []int) ([]int, error) { stop(); return v, nil }
	if _, err := loader.Next(ctx); !errors.Is(err, context.Canceled) || sampler.State().Position != 4 {
		t.Fatal("collation cancellation committed input")
	}
	loader.collate = func(v []int) ([]int, error) { panic("collate panic") }
	panicked := false
	func() { defer func() { panicked = recover() != nil }(); loader.Next(context.Background()) }()
	if !panicked || sampler.State().Position != 4 {
		t.Fatal("collation panic was not propagated transactionally")
	}
	loader.collate = func(v []int) ([]int, error) { return v, nil }
	last, err := loader.Next(context.Background())
	if err != nil || !slices.Equal(last, []int{4, 5}) {
		t.Fatal("retry after collation panic skipped input")
	}
}
