package autograd

import "fmt"

// Reduction selects how a loss combines its per-element values.
type Reduction int

const (
	// ReductionMean divides the weighted values by the sum of the weights, or by
	// the element count when there is no weight. This is the zero value.
	ReductionMean Reduction = iota
	// ReductionSum adds the weighted values without dividing.
	ReductionSum
)

// LossOptions configures the elementwise losses. Weight is an optional
// per-element weight with the prediction's shape and device: it scales each
// element, and a zero weight ignores that element for ReductionMean because the
// divisor is the sum of weights. A nil Weight counts every element once.
//
// The losses return a scalar, so ReductionNone (an unreduced tensor) is not
// available; use elementwise operators when per-element values are needed.
//
// Weighted ReductionMean follows PyTorch's per-loss divisor, which is not
// uniform: MSELoss and L1Loss divide by the sum of weights, while
// BCEWithLogitsLoss divides by the element count. Zero weights therefore leave
// the divisor of a BCE mean alone; use ReductionSum with your own divisor when
// they must not count.
type LossOptions struct {
	Reduction Reduction
	Weight    *Tensor
}

// lossAxes lists every axis of a shape so a reduction covers all elements. Sum
// and Mean with no axes reduce nothing, which is why the axes are explicit.
func lossAxes(shape []int) []int {
	axes := make([]int, len(shape))
	for i := range axes {
		axes[i] = i
	}
	return axes
}

// reduceLoss applies the reduction. divideByWeightSum selects the mean divisor:
// true divides by the sum of weights, false by the element count, matching the
// reference implementation of each loss.
func reduceLoss(elementwise *Tensor, options LossOptions, divideByWeightSum bool) *Tensor {
	if options.Reduction != ReductionMean && options.Reduction != ReductionSum {
		panic("autograd: invalid loss reduction")
	}
	weighted := elementwise
	if options.Weight != nil {
		same(elementwise, options.Weight)
		weighted = Mul(elementwise, options.Weight)
	}
	axes := lossAxes(weighted.Shape)
	if options.Reduction == ReductionSum {
		return Sum(weighted, axes...)
	}
	if options.Weight != nil && divideByWeightSum {
		return Div(Sum(weighted, axes...), Sum(options.Weight, lossAxes(options.Weight.Shape)...))
	}
	return Mean(weighted, axes...)
}

// MSELoss is the mean or sum of squared differences. It is the generic
// replacement for composing Sub/Mul/Mean per example.
func MSELoss(prediction, target *Tensor, options LossOptions) *Tensor {
	dispatchBackend("mse_loss", prediction.Device)
	same(prediction, target)
	difference := Sub(prediction, target)
	return reduceLoss(Mul(difference, difference), options, true)
}

// L1Loss is the mean or sum of absolute differences.
func L1Loss(prediction, target *Tensor, options LossOptions) *Tensor {
	dispatchBackend("l1_loss", prediction.Device)
	same(prediction, target)
	return reduceLoss(Abs(Sub(prediction, target)), options, true)
}

// BCEWithLogitsLoss is the mean or sum of the binary cross-entropy of logits,
// using the max(x,0) - x*y + log(1+exp(-|x|)) form so large magnitudes stay
// finite. Targets are probabilities in [0,1]. Use it instead of computing
// log(sigmoid(x)) yourself; a probability-space BCE has to clamp its logarithm.
func BCEWithLogitsLoss(logits, target *Tensor, options LossOptions) *Tensor {
	dispatchBackend("bce_logits_loss", logits.Device)
	same(logits, target)
	// max(x,0) - x*y
	term := Sub(ReLU(logits), Mul(logits, target))
	// log(1 + exp(-|x|))
	softplus := Log(AddScalar(Exp(MulScalar(Abs(logits), -1)), 1))
	return reduceLoss(Add(term, softplus), options, false)
}

// LossReduction names the reduction for error messages and diagnostics.
func (r Reduction) String() string {
	switch r {
	case ReductionMean:
		return "mean"
	case ReductionSum:
		return "sum"
	default:
		return fmt.Sprintf("unknown(%d)", int(r))
	}
}
