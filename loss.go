package gograd

import (
	"math"
	"sort"
)

// smoothL1Beta is the transition point of the Smooth L1 loss, matching
// torch.nn.functional.smooth_l1_loss's default beta of 1.0.
const smoothL1Beta = 1.0

// CenterRows subtracts the per-row median of the masked elements from every
// element of the row. The median is the lower middle element for an even
// count, and its gradient is shared equally among masked elements equal to the
// median, matching torch.median.
func CenterRows(values *Tensor, mask [][]bool) *Tensor {
	if values.Dims() != 2 || len(mask) != values.Shape[0] {
		panic("gograd: CenterRows expects values[B,T] and B mask rows")
	}
	batch, time := values.Shape[0], values.Shape[1]
	centers := make([]float64, batch)
	counts := make([]int, batch)
	out := NewTensor([]int{batch, time}, make([]float64, batch*time))
	scratch := make([]float64, 0, time)
	for row := 0; row < batch; row++ {
		if len(mask[row]) != time {
			panic("gograd: CenterRows mask width mismatch")
		}
		scratch = scratch[:0]
		for t := 0; t < time; t++ {
			if mask[row][t] {
				scratch = append(scratch, values.Data[row*time+t])
			}
		}
		if len(scratch) == 0 {
			continue
		}
		sort.Float64s(scratch)
		center := scratch[(len(scratch)-1)/2]
		count := 0
		for _, value := range scratch {
			if value == center {
				count++
			}
		}
		centers[row] = center
		counts[row] = count
		for t := 0; t < time; t++ {
			out.Data[row*time+t] = values.Data[row*time+t] - center
		}
	}
	out.parents = []*Tensor{values}
	out.backward = func() {
		for row := 0; row < batch; row++ {
			if counts[row] == 0 {
				for t := 0; t < time; t++ {
					values.Grad[row*time+t] += out.Grad[row*time+t]
				}
				continue
			}
			gradSum := 0.0
			for t := 0; t < time; t++ {
				gradSum += out.Grad[row*time+t]
			}
			share := gradSum / float64(counts[row])
			for t := 0; t < time; t++ {
				g := out.Grad[row*time+t]
				if mask[row][t] && values.Data[row*time+t] == centers[row] {
					g -= share
				}
				values.Grad[row*time+t] += g
			}
		}
	}
	return out
}

// SmoothL1Mean returns the mean Smooth L1 loss between two equal-shaped
// tensors.
func SmoothL1Mean(predicted, target *Tensor) *Tensor {
	if !sameShape(predicted.Shape, target.Shape) {
		panic("gograd: SmoothL1Mean requires equal shapes")
	}
	n := predicted.Numel()
	out := NewTensor([]int{1}, []float64{0})
	if n == 0 {
		out.parents = []*Tensor{predicted, target}
		out.backward = func() {}
		return out
	}
	perElement := make([]float64, n)
	total := 0.0
	for i := 0; i < n; i++ {
		d := predicted.Data[i] - target.Data[i]
		ad := math.Abs(d)
		if ad < smoothL1Beta {
			total += 0.5 * d * d / smoothL1Beta
			perElement[i] = d / smoothL1Beta
		} else {
			total += ad - 0.5*smoothL1Beta
			if d < 0 {
				perElement[i] = -1
			} else {
				perElement[i] = 1
			}
		}
	}
	out.Data[0] = total / float64(n)
	out.parents = []*Tensor{predicted, target}
	out.backward = func() {
		scale := out.Grad[0] / float64(n)
		for i := 0; i < n; i++ {
			predicted.Grad[i] += perElement[i] * scale
			target.Grad[i] -= perElement[i] * scale
		}
	}
	return out
}

// SequenceLossOptions configures SequenceLoss.
type SequenceLossOptions struct {
	// LowCents and HighCents bound the centered target when Bounded is true.
	LowCents  float64
	HighCents float64
	Bounded   bool
	// TargetScale divides the bound before clamping, matching the normalized
	// target space.
	TargetScale float64
	DeltaWeight float64
}

// SequenceLoss computes the masked Smooth L1 loss on centered frames plus a
// weighted Smooth L1 loss on adjacent-frame differences.
func SequenceLoss(predicted, target *Tensor, mask [][]bool, options SequenceLossOptions) *Tensor {
	flat := FlattenBool(mask)
	any := false
	for _, keep := range flat {
		if keep {
			any = true
			break
		}
	}
	if !any {
		return Zeros(1)
	}
	centeredPredicted := CenterRows(predicted, mask)
	centeredTarget := CenterRows(target, mask)
	if options.Bounded {
		scale := math.Max(1.0, options.TargetScale)
		centeredTarget = Clamp(centeredTarget, options.LowCents/scale, options.HighCents/scale)
	}
	absolute := SmoothL1Mean(GatherMask(centeredPredicted, flat), GatherMask(centeredTarget, flat))

	pairMask := make([][]bool, len(mask))
	pairs := 0
	for row := range mask {
		width := len(mask[row])
		rowPairs := make([]bool, 0, width)
		if width > 1 {
			for t := 0; t < width-1; t++ {
				keep := mask[row][t] && mask[row][t+1]
				rowPairs = append(rowPairs, keep)
				if keep {
					pairs++
				}
			}
		}
		pairMask[row] = rowPairs
	}
	if pairs == 0 {
		return absolute
	}
	predictedDelta := Diff(centeredPredicted)
	targetDelta := Diff(centeredTarget)
	flatPairs := FlattenBool(pairMask)
	delta := SmoothL1Mean(GatherMask(predictedDelta, flatPairs), GatherMask(targetDelta, flatPairs))
	return Add(absolute, ScaleNum(delta, options.DeltaWeight))
}
