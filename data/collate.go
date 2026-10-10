package data

import (
	"fmt"
	"slices"
)

type FloatSample struct {
	Values []float32
	Shape  []int
	Label  int
}
type FloatBatch struct {
	Values        []float32
	Shape, Labels []int
	// Padded batches additionally contain [batch,time] validity and lengths.
	Lengths []int
	Mask    []bool
}

func shapeSize(shape []int) (int, error) {
	n := 1
	limit := int(^uint(0) >> 1)
	for _, d := range shape {
		if d < 0 || d != 0 && n > limit/d {
			return 0, fmt.Errorf("data: invalid or overflowing shape %v", shape)
		}
		n *= d
	}
	return n, nil
}
func validateFloat(sample FloatSample) error {
	n, err := shapeSize(sample.Shape)
	if err != nil {
		return err
	}
	if n != len(sample.Values) {
		return fmt.Errorf("data: values do not match shape")
	}
	return nil
}

// Stack copies equal-shaped samples, prepending the batch dimension.
func Stack(samples []FloatSample) (FloatBatch, error) {
	if len(samples) == 0 {
		return FloatBatch{}, fmt.Errorf("data: cannot stack an empty batch")
	}
	shape := append([]int{len(samples)}, samples[0].Shape...)
	n, err := shapeSize(shape)
	if err != nil {
		return FloatBatch{}, err
	}
	for _, s := range samples {
		if err := validateFloat(s); err != nil {
			return FloatBatch{}, err
		}
		if !slices.Equal(s.Shape, samples[0].Shape) {
			return FloatBatch{}, fmt.Errorf("data: stack requires equal shapes")
		}
	}
	b := FloatBatch{Values: make([]float32, n), Shape: shape, Labels: make([]int, len(samples))}
	for i, s := range samples {
		copy(b.Values[i*len(s.Values):], s.Values)
		b.Labels[i] = s.Label
	}
	return b, nil
}

type PaddingOptions struct {
	Value float32
	// Length=0 uses the batch maximum. A shorter positive length requires Truncate.
	Length   int
	Truncate bool
}

// PadSequences pads/truncates the first sample axis, preserving equal trailing
// dimensions. Mask is true only for retained input positions. Zero-length
// sequences are accepted, including an all-empty batch with time dimension zero.
func PadSequences(samples []FloatSample, options PaddingOptions) (FloatBatch, error) {
	if len(samples) == 0 || options.Length < 0 {
		return FloatBatch{}, fmt.Errorf("data: invalid padded batch")
	}
	longest := 0
	for _, s := range samples {
		if err := validateFloat(s); err != nil {
			return FloatBatch{}, err
		}
		if len(s.Shape) == 0 || !slices.Equal(s.Shape[1:], samples[0].Shape[1:]) {
			return FloatBatch{}, fmt.Errorf("data: padding requires matching trailing dimensions")
		}
		longest = max(longest, s.Shape[0])
	}
	length := options.Length
	if length == 0 {
		length = longest
	}
	if length < longest && !options.Truncate {
		return FloatBatch{}, fmt.Errorf("data: padding length would truncate samples")
	}
	shape := append([]int{len(samples), length}, samples[0].Shape[1:]...)
	n, err := shapeSize(shape)
	if err != nil {
		return FloatBatch{}, err
	}
	maskSize, err := shapeSize([]int{len(samples), length})
	if err != nil {
		return FloatBatch{}, err
	}
	width, err := shapeSize(samples[0].Shape[1:])
	if err != nil {
		return FloatBatch{}, err
	}
	b := FloatBatch{Values: make([]float32, n), Shape: shape, Labels: make([]int, len(samples)), Lengths: make([]int, len(samples)), Mask: make([]bool, maskSize)}
	if options.Value != 0 {
		for i := range b.Values {
			b.Values[i] = options.Value
		}
	}
	for i, s := range samples {
		kept := min(s.Shape[0], length)
		b.Lengths[i], b.Labels[i] = kept, s.Label
		copy(b.Values[i*length*width:], s.Values[:kept*width])
		for t := 0; t < kept; t++ {
			b.Mask[i*length+t] = true
		}
	}
	return b, nil
}
