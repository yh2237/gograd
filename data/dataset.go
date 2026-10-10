// Package data provides host-side datasets, sampling and synchronous batching.
// Tensor construction/device transfer belongs to the caller's training loop.
package data

import (
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
)

// Dataset has stable length and index identity for an iteration/checkpoint.
// Samples can borrow dataset memory; Collate must not mutate borrowed samples.
type Dataset[T any] interface {
	Len() int
	Get(context.Context, int) (T, error)
}

type SliceDataset[T any] []T

func (d SliceDataset[T]) Len() int { return len(d) }
func (d SliceDataset[T]) Get(ctx context.Context, index int) (T, error) {
	var zero T
	if ctx == nil {
		return zero, fmt.Errorf("data: nil context")
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if index < 0 || index >= len(d) {
		return zero, fmt.Errorf("data: index %d out of range", index)
	}
	return d[index], nil
}

// Subset is an index view. Constructor copies indices; duplicate indices are
// allowed when deliberate repeated examples are wanted.
type Subset[T any] struct {
	dataset Dataset[T]
	indices []int
}

func NewSubset[T any](dataset Dataset[T], indices []int) (*Subset[T], error) {
	if dataset == nil || dataset.Len() < 0 {
		return nil, fmt.Errorf("data: invalid subset dataset")
	}
	for _, i := range indices {
		if i < 0 || i >= dataset.Len() {
			return nil, fmt.Errorf("data: invalid subset index %d", i)
		}
	}
	return &Subset[T]{dataset, slices.Clone(indices)}, nil
}
func (d *Subset[T]) Len() int { return len(d.indices) }
func (d *Subset[T]) Get(ctx context.Context, index int) (T, error) {
	var zero T
	if ctx == nil {
		return zero, fmt.Errorf("data: nil context")
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if index < 0 || index >= len(d.indices) {
		return zero, fmt.Errorf("data: subset index %d out of range", index)
	}
	return d.dataset.Get(ctx, d.indices[index])
}
func (d *Subset[T]) Indices() []int { return slices.Clone(d.indices) }

// RandomSplit returns disjoint train/validation indices covering the dataset.
// Keep these indices (or the identical size/count/seed) in resumable metadata.
func RandomSplit(size, validation int, seed uint64) (train, valid []int, err error) {
	if size < 0 || validation < 0 || validation > size {
		return nil, nil, fmt.Errorf("data: invalid split size")
	}
	order := make([]int, size)
	for i := range order {
		order[i] = i
	}
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	r.Shuffle(size, func(i, j int) { order[i], order[j] = order[j], order[i] })
	return slices.Clone(order[validation:]), slices.Clone(order[:validation]), nil
}
