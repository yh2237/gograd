package data

import (
	"context"
	"fmt"
	randv2 "math/rand/v2"
)

// FlipsHorizontallyFloat32 swaps columns in each channel plane. values is in
// [channels,height,width] order; channels <= 0 defaults to 1.
func FlipsHorizontallyFloat32(values []float32, channels, height, width int) {
	if channels < 1 {
		channels = 1
	}
	for c := 0; c < channels; c++ {
		for y := 0; y < height; y++ {
			base := c*height*width + y*width
			for x := 0; x < width/2; x++ {
				left, right := base+x, base+width-1-x
				values[left], values[right] = values[right], values[left]
			}
		}
	}
}

// AddGaussianNoiseFloat32 adds deterministic Gaussian noise to values in place.
// The RNG is seeded from seed and index, so worker/collate/order changes reproduce
// the exact same numbers and resuming keeps the same samples.
func AddGaussianNoiseFloat32(values []float32, seed uint64, index int, std float32) {
	if std == 0 {
		return
	}
	rng := randv2.New(randv2.NewPCG(seed, uint64(index)+0x9e3779b97f4a7c15))
	for i := range values {
		values[i] += float32(rng.NormFloat64()) * std
	}
}

// TransformDataset applies a deterministic per-sample transform while preserving
// the underlying dataset ordering/fingerprint. seed is mixed in so different
// augmentation policies do not collide.
type TransformDataset[T any] struct {
	Base  Dataset[T]
	Seed  uint64
	Apply func(index int, seed uint64, sample T) (T, error)
}

func (d TransformDataset[T]) Len() int { return d.Base.Len() }

func (d TransformDataset[T]) Get(ctx context.Context, index int) (T, error) {
	var zero T
	sample, err := d.Base.Get(ctx, index)
	if err != nil {
		return zero, err
	}
	if d.Apply == nil {
		return sample, nil
	}
	return d.Apply(index, d.Seed, sample)
}

// FlipEarly flips the first fraction of samples deterministically. This is a small
// helper for image augmentation tests and demos.
func FlipEarly(values []float32, channels, height, width, seed, index int, probability float64) {
	if probability <= 0 {
		return
	}
	rng := randv2.New(randv2.NewPCG(uint64(seed), uint64(index)^0x9e3779b97f4a7c15))
	if rng.Float64() < probability {
		FlipsHorizontallyFloat32(values, channels, height, width)
	}
}

// Ensure fmt stays referenced if needed elsewhere.
var _ = fmt.Errorf
