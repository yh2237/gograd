package data

import (
	"context"
	"fmt"
	"math"
	randv2 "math/rand/v2"
)

// FlipsHorizontallyFloat32 swaps columns in each channel plane. values is in
// [channels,height,width] order. Invalid shapes leave values unchanged.
func FlipsHorizontallyFloat32(values []float32, channels, height, width int) error {
	if err := validateImageShape(values, channels, height, width); err != nil {
		return err
	}
	flipHorizontally(values, channels, height, width)
	return nil
}

func validateImageShape(values []float32, channels, height, width int) error {
	if channels <= 0 || height <= 0 || width <= 0 {
		return fmt.Errorf("data: positive image dimensions required")
	}
	n, err := shapeSize([]int{channels, height, width})
	if err != nil {
		return err
	}
	if n != len(values) {
		return fmt.Errorf("data: image values do not match shape")
	}
	return nil
}

func flipHorizontally(values []float32, channels, height, width int) {
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
// the exact same numbers and resuming keeps the same samples. Index must be
// nonnegative; std and values must be finite, and std must be nonnegative.
// Validation or overflow errors leave values unchanged. No clipping is applied.
func AddGaussianNoiseFloat32(values []float32, seed uint64, index int, std float32) error {
	if index < 0 || std < 0 || math.IsNaN(float64(std)) || math.IsInf(float64(std), 0) {
		return fmt.Errorf("data: invalid noise index or standard deviation")
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("data: noise input must be finite")
		}
	}
	if std == 0 {
		return nil
	}
	rng := randv2.New(randv2.NewPCG(seed, uint64(index)+0x9e3779b97f4a7c15))
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = value + float32(rng.NormFloat64())*std
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			return fmt.Errorf("data: noise output overflow")
		}
	}
	copy(values, out)
	return nil
}

// TransformDataset passes the dataset index and Seed unchanged to Apply. It does
// not mix seeds, copy samples, forward fingerprints or checkpoint transform state.
// Apply must copy borrowed slices before mutation and be deterministic for the
// same seed/index/sample. With workers, Base.Get and Apply must support concurrent
// calls; they can also be retried or discarded by prefetch. Keep policy/version,
// seed and base-data identity in application checkpoint metadata. Epochs do not
// change augmentation. Wrap a base dataset before Subset to keep original indices.
type TransformDataset[T any] struct {
	Base  Dataset[T]
	Seed  uint64
	Apply func(index int, seed uint64, sample T) (T, error)
}

func (d TransformDataset[T]) Len() int {
	if d.Base == nil {
		return 0
	}
	return d.Base.Len()
}

func (d TransformDataset[T]) Get(ctx context.Context, index int) (T, error) {
	var zero T
	if ctx == nil {
		return zero, fmt.Errorf("data: nil context")
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if d.Base == nil || index < 0 || index >= d.Base.Len() {
		return zero, fmt.Errorf("data: invalid transform dataset or index")
	}
	sample, err := d.Base.Get(ctx, index)
	if err != nil {
		return zero, err
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if d.Apply != nil {
		sample, err = d.Apply(index, d.Seed, sample)
		if err != nil {
			return zero, err
		}
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	return sample, nil
}

// RandomHorizontalFlipFloat32 flips a CHW image with the given probability using
// only seed/index. Probability must be finite and in [0,1]; index is nonnegative.
// All validation runs even when probability is zero. Errors do not mutate values.
func RandomHorizontalFlipFloat32(values []float32, channels, height, width int, seed uint64, index int, probability float64) error {
	if err := validateImageShape(values, channels, height, width); err != nil {
		return err
	}
	if index < 0 || probability < 0 || probability > 1 || math.IsNaN(probability) || math.IsInf(probability, 0) {
		return fmt.Errorf("data: invalid flip index or probability")
	}
	rng := randv2.New(randv2.NewPCG(seed, uint64(index)^0x9e3779b97f4a7c15))
	if rng.Float64() < probability {
		flipHorizontally(values, channels, height, width)
	}
	return nil
}

// FlipEarly is the legacy int-seed spelling of RandomHorizontalFlipFloat32.
// New callers should use the uint64-seed API to avoid truncation on 32-bit hosts.
func FlipEarly(values []float32, channels, height, width, seed, index int, probability float64) error {
	return RandomHorizontalFlipFloat32(values, channels, height, width, uint64(seed), index, probability)
}
