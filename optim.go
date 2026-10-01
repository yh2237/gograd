package gograd

import "math"

// AdamW implements the decoupled-weight-decay Adam update. Its defaults match
// torch.optim.AdamW.
type AdamW struct {
	params      []*Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	step        int
	m           [][]float64
	v           [][]float64
}

// NewAdamW returns an optimizer over params with the given learning rate and
// weight decay.
func NewAdamW(params []*Tensor, learningRate, weightDecay float64) *AdamW {
	optimizer := &AdamW{
		params:      params,
		lr:          learningRate,
		beta1:       0.9,
		beta2:       0.999,
		eps:         1e-8,
		weightDecay: weightDecay,
	}
	for _, param := range params {
		optimizer.m = append(optimizer.m, make([]float64, param.Numel()))
		optimizer.v = append(optimizer.v, make([]float64, param.Numel()))
	}
	return optimizer
}

// ZeroGrad clears every parameter gradient.
func (o *AdamW) ZeroGrad() {
	for _, param := range o.params {
		param.ZeroGrad()
	}
}

// Step applies one AdamW update using the current gradients.
func (o *AdamW) Step() {
	o.step++
	biasCorrection1 := 1 - math.Pow(o.beta1, float64(o.step))
	biasCorrection2 := 1 - math.Pow(o.beta2, float64(o.step))
	stepSize := o.lr / biasCorrection1
	for index, param := range o.params {
		m := o.m[index]
		v := o.v[index]
		for i := range param.Data {
			g := param.Grad[i]
			m[i] = o.beta1*m[i] + (1-o.beta1)*g
			v[i] = o.beta2*v[i] + (1-o.beta2)*g*g
			denom := math.Sqrt(v[i])/math.Sqrt(biasCorrection2) + o.eps
			param.Data[i] *= 1 - o.lr*o.weightDecay
			param.Data[i] -= stepSize * m[i] / denom
		}
	}
}

// ClipGradNorm scales all gradients so that their global L2 norm is at most
// maxNorm, matching torch.nn.utils.clip_grad_norm_ with the default 2-norm.
func ClipGradNorm(params []*Tensor, maxNorm float64) float64 {
	total := 0.0
	for _, param := range params {
		for _, g := range param.Grad {
			total += g * g
		}
	}
	total = math.Sqrt(total)
	coefficient := maxNorm / (total + 1e-6)
	if coefficient < 1 {
		for _, param := range params {
			for i := range param.Grad {
				param.Grad[i] *= coefficient
			}
		}
	}
	return total
}
