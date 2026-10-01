package gputcn

import (
	"math"
	"sort"
)

// LossOptions configures LossGrad. It mirrors the CPU SequenceLoss options.
type LossOptions struct {
	LowCents    float64
	HighCents   float64
	Bounded     bool
	TargetScale float64
	DeltaWeight float64
}

// LossGrad computes the masked Smooth L1 loss over centered frames plus the
// weighted adjacent-frame delta loss, and returns the loss and its gradient
// with respect to predicted. It follows the CPU SequenceLoss but works on flat
// slices without building an autograd graph.
func LossGrad(predicted []float32, target []float64, mask [][]bool, options LossOptions) (float64, []float32) {
	rows := len(mask)
	time := 0
	if rows > 0 {
		time = len(mask[0])
	}
	centeredPredicted := make([]float64, rows*time)
	centeredTarget := make([]float64, rows*time)
	medians := make([]float64, rows)
	targetMedians := make([]float64, rows)
	counts := make([]int, rows)
	scratch := make([]float64, 0, time)
	for row := 0; row < rows; row++ {
		scratch = scratch[:0]
		for t := 0; t < time; t++ {
			if mask[row][t] {
				scratch = append(scratch, float64(predicted[row*time+t]))
			}
		}
		if len(scratch) > 0 {
			sort.Float64s(scratch)
			median := scratch[(len(scratch)-1)/2]
			medians[row] = median
			for _, value := range scratch {
				if value == median {
					counts[row]++
				}
			}
		}
		scratch = scratch[:0]
		for t := 0; t < time; t++ {
			if mask[row][t] {
				scratch = append(scratch, target[row*time+t])
			}
		}
		if len(scratch) > 0 {
			sort.Float64s(scratch)
			targetMedians[row] = scratch[(len(scratch)-1)/2]
		}
		for t := 0; t < time; t++ {
			centeredPredicted[row*time+t] = float64(predicted[row*time+t]) - medians[row]
			centeredTarget[row*time+t] = target[row*time+t] - targetMedians[row]
		}
	}
	if options.Bounded {
		scale := math.Max(1, options.TargetScale)
		low, high := options.LowCents/scale, options.HighCents/scale
		for i := range centeredTarget {
			if centeredTarget[i] < low {
				centeredTarget[i] = low
			}
			if centeredTarget[i] > high {
				centeredTarget[i] = high
			}
		}
	}

	gradient := make([]float64, rows*time)
	valid := 0
	for row := 0; row < rows; row++ {
		for t := 0; t < time; t++ {
			if mask[row][t] {
				valid++
			}
		}
	}
	absoluteLoss := 0.0
	if valid > 0 {
		for row := 0; row < rows; row++ {
			for t := 0; t < time; t++ {
				if !mask[row][t] {
					continue
				}
				difference := centeredPredicted[row*time+t] - centeredTarget[row*time+t]
				absoluteLoss += smoothL1(difference, &gradient[row*time+t])
			}
		}
		absoluteLoss /= float64(valid)
		for i := range gradient {
			gradient[i] /= float64(valid)
		}
	}

	pairs := 0
	deltaGradient := make([]float64, rows*time)
	deltaLoss := 0.0
	for row := 0; row < rows; row++ {
		for t := 1; t < time; t++ {
			if !(mask[row][t-1] && mask[row][t]) {
				continue
			}
			difference := (centeredPredicted[row*time+t] - centeredPredicted[row*time+t-1]) -
				(centeredTarget[row*time+t] - centeredTarget[row*time+t-1])
			pairs++
			deltaLoss += smoothL1(difference, &deltaGradient[row*time+t])
			deltaGradient[row*time+t-1] -= deltaGradient[row*time+t]
		}
	}
	if pairs > 0 {
		deltaLoss /= float64(pairs)
		for i := range deltaGradient {
			gradient[i] += options.DeltaWeight * deltaGradient[i] / float64(pairs)
		}
	}

	result := make([]float32, len(predicted))
	for row := 0; row < rows; row++ {
		if counts[row] == 0 {
			for t := 0; t < time; t++ {
				result[row*time+t] = float32(gradient[row*time+t])
			}
			continue
		}
		sum := 0.0
		for t := 0; t < time; t++ {
			sum += gradient[row*time+t]
		}
		share := sum / float64(counts[row])
		for t := 0; t < time; t++ {
			value := gradient[row*time+t]
			if mask[row][t] && float64(predicted[row*time+t]) == medians[row] {
				value -= share
			}
			result[row*time+t] = float32(value)
		}
	}
	return absoluteLoss + options.DeltaWeight*deltaLoss, result
}

// smoothL1 adds the Smooth L1 loss for difference to grad and returns the
// loss term. Beta is 1.
func smoothL1(difference float64, grad *float64) float64 {
	absolute := math.Abs(difference)
	if absolute < 1 {
		*grad += difference
		return 0.5 * difference * difference
	}
	if difference < 0 {
		*grad -= 1
	} else {
		*grad += 1
	}
	return absolute - 0.5
}
