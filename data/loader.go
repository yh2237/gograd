package data

import (
	"context"
	"errors"
	"fmt"
	"io"
)

type CollateFunc[T, B any] func([]T) (B, error)
type LoaderOptions struct {
	BatchSize int
	DropLast  bool
}
type Loader[T, B any] struct {
	dataset Dataset[T]
	sampler Sampler
	collate CollateFunc[T, B]
	options LoaderOptions
	size    int
}
type LoaderState struct {
	Version, DatasetSize, BatchSize int
	DropLast                        bool
	Sampler                         []byte
}

func NewLoader[T, B any](dataset Dataset[T], sampler Sampler, collate CollateFunc[T, B], options LoaderOptions) (*Loader[T, B], error) {
	if dataset == nil || sampler == nil || collate == nil || options.BatchSize < 1 || dataset.Len() < 0 || sampler.Len() != dataset.Len() {
		return nil, fmt.Errorf("data: invalid loader configuration")
	}
	return &Loader[T, B]{dataset, sampler, collate, options, dataset.Len()}, nil
}

// Next returns io.EOF only at the end of this epoch. Read/collate/cancellation
// errors leave the cursor unchanged for a retry; callback EOF becomes UnexpectedEOF.
// The caller owns a successful batch; builtin float collates return fresh slices.
func (l *Loader[T, B]) Next(ctx context.Context) (B, error) {
	var zero B
	if ctx == nil {
		return zero, fmt.Errorf("data: nil context")
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if l.dataset.Len() != l.size || l.sampler.Len() != l.size {
		return zero, fmt.Errorf("data: dataset length changed")
	}
	indices, err := l.sampler.Peek(l.options.BatchSize)
	if err != nil {
		return zero, err
	}
	if len(indices) == 0 || len(indices) > l.options.BatchSize {
		return zero, fmt.Errorf("data: invalid sampler batch")
	}
	if l.options.DropLast && len(indices) < l.options.BatchSize {
		if err := l.sampler.Advance(len(indices)); err != nil {
			return zero, callbackError("sampler advance", err)
		}
		return zero, io.EOF
	}
	samples := make([]T, len(indices))
	for i, index := range indices {
		if err := ctx.Err(); err != nil {
			return zero, err
		}
		if index < 0 || index >= l.size {
			return zero, fmt.Errorf("data: sampler index %d out of range", index)
		}
		samples[i], err = l.dataset.Get(ctx, index)
		if err != nil {
			return zero, callbackError(fmt.Sprintf("sample %d", index), err)
		}
	}
	batch, err := l.collate(samples)
	if err != nil {
		return zero, callbackError("collate", err)
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if err := l.sampler.Advance(len(indices)); err != nil {
		return zero, callbackError("sampler advance", err)
	}
	return batch, nil
}
func callbackError(operation string, err error) error {
	if errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF
	}
	return fmt.Errorf("data: %s: %w", operation, err)
}
func (l *Loader[T, B]) NextEpoch() error { return l.sampler.NextEpoch() }
func (l *Loader[T, B]) State() (LoaderState, error) {
	s, ok := l.sampler.(StatefulSampler)
	if !ok {
		return LoaderState{}, fmt.Errorf("data: sampler does not support checkpoints")
	}
	payload, err := s.MarshalBinary()
	if err != nil {
		return LoaderState{}, err
	}
	return LoaderState{1, l.size, l.options.BatchSize, l.options.DropLast, payload}, nil
}
func (l *Loader[T, B]) LoadState(state LoaderState) error {
	if state.Version != 1 || state.DatasetSize != l.size || state.BatchSize != l.options.BatchSize || state.DropLast != l.options.DropLast || l.dataset.Len() != l.size {
		return fmt.Errorf("data: loader state does not match configuration")
	}
	s, ok := l.sampler.(StatefulSampler)
	if !ok {
		return fmt.Errorf("data: sampler does not support checkpoints")
	}
	return s.UnmarshalBinary(state.Sampler)
}
