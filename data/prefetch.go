package data

import (
	"context"
	"fmt"
	"io"
	"sync"
)

type sampleJob struct{ batch, slot, index int }
type sampleResult[T any] struct {
	batch, slot int
	value       T
	err         error
}
type pendingBatch[T any] struct {
	id        int
	indices   []int
	samples   []T
	errors    []error
	remaining int
	drop      bool
}
type prefetchPipeline[T any] struct {
	ctx       context.Context
	cancel    context.CancelFunc
	jobs      chan sampleJob
	results   chan sampleResult[T]
	workers   sync.WaitGroup
	pending   []*pendingBatch[T]
	nextID    int
	ended     bool
	planError error
}

func (l *Loader[T, B]) startPipeline(ctx context.Context) {
	readContext, cancel := context.WithCancel(ctx)
	capacity := min(l.size, l.options.Prefetch*l.options.BatchSize)
	p := &prefetchPipeline[T]{ctx: readContext, cancel: cancel, jobs: make(chan sampleJob, capacity), results: make(chan sampleResult[T], capacity)}
	l.pipeline = p
	for i := 0; i < min(l.options.Workers, capacity); i++ {
		p.workers.Add(1)
		go func() {
			defer p.workers.Done()
			for {
				select {
				case <-p.ctx.Done():
					return
				case job := <-p.jobs:
					if p.ctx.Err() != nil {
						return
					}
					value, err := readSample(p.ctx, l.dataset, job.index)
					result := sampleResult[T]{job.batch, job.slot, value, err}
					select {
					case p.results <- result:
					case <-p.ctx.Done():
						return
					}
				}
			}
		}()
	}
}

// A callback panic becomes an ordered read error rather than killing a worker
// and leaving Next waiting forever. A failed batch is retried from its start.
func readSample[T any](ctx context.Context, dataset Dataset[T], index int) (value T, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("data: sample %d panicked: %v", index, recovered)
		}
	}()
	value, err = dataset.Get(ctx, index)
	if err != nil {
		err = callbackError(fmt.Sprintf("sample %d", index), err)
	}
	return
}

func (l *Loader[T, B]) stopPipeline() {
	if l.pipeline == nil {
		return
	}
	p := l.pipeline
	p.cancel()
	p.workers.Wait()
	l.pipeline = nil
}

// Only the consumer touches pending batches and the sampler. Workers exchange
// immutable indices/results through bounded channels and never advance state.
func (l *Loader[T, B]) fillPipeline() {
	p := l.pipeline
	planned := 0
	for _, batch := range p.pending {
		planned += len(batch.indices)
	}
	for len(p.pending) < l.options.Prefetch && !p.ended {
		prefix, err := l.sampler.Peek(planned + l.options.BatchSize)
		if err != nil {
			p.ended = true
			if err != io.EOF {
				p.planError = err
			}
			return
		}
		if len(prefix) < planned || len(prefix) > planned+l.options.BatchSize {
			p.ended = true
			p.planError = fmt.Errorf("data: invalid sampler lookahead")
			return
		}
		if len(prefix) == planned {
			p.ended = true
			return
		}
		indices := append([]int(nil), prefix[planned:]...)
		batch := &pendingBatch[T]{id: p.nextID, indices: indices, samples: make([]T, len(indices)), errors: make([]error, len(indices))}
		p.nextID++
		p.pending = append(p.pending, batch)
		planned += len(indices)
		if l.options.DropLast && len(indices) < l.options.BatchSize {
			batch.drop = true
			p.ended = true
			return
		}
		for slot, index := range indices {
			if index < 0 || index >= l.size {
				batch.errors[slot] = fmt.Errorf("data: sampler index %d out of range", index)
				continue
			}
			batch.remaining++
			select {
			case p.jobs <- sampleJob{batch.id, slot, index}:
			case <-p.ctx.Done():
				return
			}
		}
	}
}

func (l *Loader[T, B]) acceptResult(result sampleResult[T]) {
	for _, batch := range l.pipeline.pending {
		if batch.id != result.batch {
			continue
		}
		batch.samples[result.slot], batch.errors[result.slot] = result.value, result.err
		batch.remaining--
		return
	}
}

func (l *Loader[T, B]) nextPrefetched(ctx context.Context) (B, error) {
	var zero B
	defer func() {
		if failure := recover(); failure != nil {
			l.stopPipeline()
			panic(failure)
		}
	}()
	// A cancelled earlier Next context invalidates its lookahead, not the
	// committed sampler. A new live context can restart at the same position.
	if l.pipeline != nil && l.pipeline.ctx.Err() != nil {
		l.stopPipeline()
	}
	if l.pipeline == nil {
		l.startPipeline(ctx)
	}
	l.fillPipeline()
	p := l.pipeline
	if len(p.pending) == 0 {
		err := p.planError
		if err == nil {
			err = io.EOF
		}
		l.stopPipeline()
		return zero, err
	}
	batch := p.pending[0]
	for batch.remaining > 0 {
		select {
		case <-ctx.Done():
			l.stopPipeline()
			return zero, ctx.Err()
		case <-p.ctx.Done():
			err := p.ctx.Err()
			l.stopPipeline()
			return zero, err
		case result := <-p.results:
			l.acceptResult(result)
		}
	}
	if err := ctx.Err(); err != nil {
		l.stopPipeline()
		return zero, err
	}
	if err := p.ctx.Err(); err != nil {
		l.stopPipeline()
		return zero, err
	}
	for _, err := range batch.errors {
		if err != nil {
			l.stopPipeline()
			return zero, err
		}
	}
	if batch.drop {
		err := l.sampler.Advance(len(batch.indices))
		l.stopPipeline()
		if err != nil {
			return zero, callbackError("sampler advance", err)
		}
		return zero, io.EOF
	}
	out, err := l.collate(batch.samples)
	if err != nil {
		l.stopPipeline()
		return zero, callbackError("collate", err)
	}
	if err := ctx.Err(); err != nil {
		l.stopPipeline()
		return zero, err
	}
	if err := p.ctx.Err(); err != nil {
		l.stopPipeline()
		return zero, err
	}
	if err := l.sampler.Advance(len(batch.indices)); err != nil {
		l.stopPipeline()
		return zero, callbackError("sampler advance", err)
	}
	// Drop references to delivered samples before reserving the next lookahead.
	p.pending[0] = nil
	p.pending = p.pending[1:]
	l.fillPipeline()
	if len(p.pending) == 0 && p.ended && p.planError == nil {
		l.stopPipeline()
	}
	return out, nil
}
