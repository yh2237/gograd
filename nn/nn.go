// Package nn builds small trainable models from GPU tensors. Each module has
// an explicit forward and backward pass and exposes its parameters and
// gradients, so the same optimizer and training loop work for any model.
package nn

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"github.com/yh2237/gograd/tensor"
)

// Module is a trainable layer.
type Module interface {
	Parameters() []*tensor.Tensor
	Gradients() []*tensor.Tensor
	ZeroGrad()
	Forward(blas *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error)
	Backward(blas *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error)
}

// Linear computes y = x*W^T + b for x [rows,in].
type Linear struct {
	Weight *tensor.Tensor // [out,in]
	Bias   *tensor.Tensor // [out]

	gradWeight *tensor.Tensor
	gradBias   *tensor.Tensor
	input      *tensor.Tensor
	rows       int
	in         int
	out        int
}

// NewLinear creates a Linear with PyTorch-like uniform initialization.
func NewLinear(in, out int, rng *rand.Rand) (*Linear, error) {
	return NewLinearOn(tensor.CUDA, in, out, rng)
}

// NewLinearOn creates a Linear on device.
func NewLinearOn(device tensor.Device, in, out int, rng *rand.Rand) (*Linear, error) {
	weight := make([]float32, out*in)
	bound := 1 / math.Sqrt(float64(in))
	for i := range weight {
		weight[i] = float32((rng.Float64()*2 - 1) * bound)
	}
	w, err := tensor.FromHostOn(device, []int{out, in}, weight)
	if err != nil {
		return nil, err
	}
	b, err := tensor.FromHostOn(device, []int{out}, make([]float32, out))
	if err != nil {
		w.Close()
		return nil, err
	}
	gw, err := tensor.ZerosOn(device, out, in)
	if err != nil {
		w.Close()
		b.Close()
		return nil, err
	}
	gb, err := tensor.ZerosOn(device, out)
	if err != nil {
		w.Close()
		b.Close()
		gw.Close()
		return nil, err
	}
	return &Linear{Weight: w, Bias: b, gradWeight: gw, gradBias: gb, in: in, out: out}, nil
}

// Parameters returns the trainable tensors.
func (l *Linear) Parameters() []*tensor.Tensor { return []*tensor.Tensor{l.Weight, l.Bias} }

// Gradients returns the matching gradient tensors.
func (l *Linear) Gradients() []*tensor.Tensor { return []*tensor.Tensor{l.gradWeight, l.gradBias} }

// ZeroGrad clears the accumulated gradients.
func (l *Linear) ZeroGrad() {
	l.gradWeight.Zero()
	l.gradBias.Zero()
}

// Close releases the layer tensors.
func (l *Linear) Close() {
	l.Weight.Close()
	l.Bias.Close()
	l.gradWeight.Close()
	l.gradBias.Close()
}

// Forward computes y = x*W^T + b.
func (l *Linear) Forward(blas *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if (len(shape) != 2 && len(shape) != 3) || shape[len(shape)-1] != l.in || x.Device() != l.Weight.Device() {
		return nil, fmt.Errorf("nn: Linear expects [rows,%d], got %v", l.in, shape)
	}
	rows := x.Numel() / l.in
	outShape := append([]int(nil), shape...)
	outShape[len(outShape)-1] = l.out
	y, err := tensor.NewOn(x.Device(), outShape...)
	if err != nil {
		return nil, err
	}
	if x.Device() == tensor.CPU {
		w := make([]float32, l.in*l.out)
		for o := 0; o < l.out; o++ {
			for i := 0; i < l.in; i++ {
				w[i*l.out+o] = l.Weight.Data()[o*l.in+i]
			}
		}
		tensor.SGEMM(y.Data(), x.Data(), w, rows, l.out, l.in)
		for r := 0; r < rows; r++ {
			for o := 0; o < l.out; o++ {
				y.Data()[r*l.out+o] += l.Bias.Data()[o]
			}
		}
	} else if err := blas.SgemmRowMajorNT(rows, l.out, l.in, 1, x.Buffer().Pointer(), l.in, l.Weight.Buffer().Pointer(), l.in, 0, y.Buffer().Pointer(), l.out); err != nil {
		y.Close()
		return nil, err
	}
	if x.Device() == tensor.CUDA {
		if err := kernels.AddBiasColumns(y.Buffer(), l.Bias.Buffer(), rows, l.out); err != nil {
			y.Close()
			return nil, err
		}
	}
	l.input = x
	l.rows = rows
	return y, nil
}

// Backward accumulates dW and db and returns dX.
func (l *Linear) Backward(blas *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error) {
	shape := l.input.Shape()
	dx, err := tensor.NewOn(grad.Device(), shape...)
	if err != nil {
		return nil, err
	}
	if grad.Device() == tensor.CPU {
		tensor.SGEMM(dx.Data(), grad.Data(), l.Weight.Data(), l.rows, l.in, l.out)
		// dW = dY^T X; parameter gradients accumulate until ZeroGrad.
		for r := 0; r < l.rows; r++ {
			for o := 0; o < l.out; o++ {
				g := grad.Data()[r*l.out+o]
				l.gradBias.Data()[o] += g
				for i := 0; i < l.in; i++ {
					l.gradWeight.Data()[o*l.in+i] += g * l.input.Data()[r*l.in+i]
				}
			}
		}
		return dx, nil
	}
	if err := blas.SgemmRowMajor(l.rows, l.in, l.out, 1, grad.Buffer().Pointer(), l.out, l.Weight.Buffer().Pointer(), l.in, 0, dx.Buffer().Pointer(), l.in); err != nil {
		dx.Close()
		return nil, err
	}
	if err := blas.SgemmRowMajorTransposeA(l.out, l.in, l.rows, 1, grad.Buffer().Pointer(), l.out, l.input.Buffer().Pointer(), l.in, 0, l.gradWeight.Buffer().Pointer(), l.in); err != nil {
		dx.Close()
		return nil, err
	}
	if err := kernels.ColumnSum(grad.Buffer(), l.gradBias.Buffer(), l.rows, l.out); err != nil {
		dx.Close()
		return nil, err
	}
	return dx, nil
}

// Tanh applies the elementwise hyperbolic tangent.
type Tanh struct {
	output *tensor.Tensor
}

// Parameters returns nil.
func (t *Tanh) Parameters() []*tensor.Tensor { return nil }

// Gradients returns nil.
func (t *Tanh) Gradients() []*tensor.Tensor { return nil }

// ZeroGrad does nothing.
func (t *Tanh) ZeroGrad() {}

// Forward computes tanh(x).
func (t *Tanh) Forward(blas *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := tensor.NewOn(x.Device(), x.Shape()...)
	if err != nil {
		return nil, err
	}
	if x.Device() == tensor.CPU {
		for i, v := range x.Data() {
			out.Data()[i] = float32(math.Tanh(float64(v)))
		}
	} else if err := kernels.Tanh(x.Buffer(), out.Buffer(), x.Numel()); err != nil {
		out.Close()
		return nil, err
	}
	t.output = out
	return out, nil
}

// Backward returns dX = grad * (1 - out^2).
func (t *Tanh) Backward(blas *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error) {
	dx, err := tensor.ZerosOn(grad.Device(), grad.Shape()...)
	if err != nil {
		return nil, err
	}
	if grad.Device() == tensor.CPU {
		for i, v := range t.output.Data() {
			dx.Data()[i] = grad.Data()[i] * (1 - v*v)
		}
	} else if err := kernels.TanhBackward(t.output.Buffer(), grad.Buffer(), dx.Buffer(), grad.Numel()); err != nil {
		dx.Close()
		return nil, err
	}
	return dx, nil
}

// Sequential chains modules in order.
type Sequential struct {
	Modules []Module

	saved  []*tensor.Tensor
	output *tensor.Tensor
}

// Parameters returns the parameters of every module.
func (s *Sequential) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	for _, m := range s.Modules {
		params = append(params, m.Parameters()...)
	}
	return params
}

// Gradients returns the gradients of every module.
func (s *Sequential) Gradients() []*tensor.Tensor {
	var grads []*tensor.Tensor
	for _, m := range s.Modules {
		grads = append(grads, m.Gradients()...)
	}
	return grads
}

// ZeroGrad clears every module's gradients.
func (s *Sequential) ZeroGrad() {
	for _, m := range s.Modules {
		m.ZeroGrad()
	}
}

// Forward runs the modules in order.
func (s *Sequential) Forward(blas *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	s.saved = s.saved[:0]
	current := x
	for _, m := range s.Modules {
		next, err := m.Forward(blas, current)
		if err != nil {
			return nil, err
		}
		if current != x {
			s.saved = append(s.saved, current)
		}
		current = next
	}
	s.output = current
	return current, nil
}

// Backward runs the modules in reverse and returns the input gradient.
func (s *Sequential) Backward(blas *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error) {
	current := grad
	for i := len(s.Modules) - 1; i >= 0; i-- {
		next, err := s.Modules[i].Backward(blas, current)
		if err != nil {
			return nil, err
		}
		if current != grad {
			current.Close()
		}
		current = next
	}
	return current, nil
}

// CloseActivations releases the tensors saved by the last forward pass. Call it
// after Backward.
func (s *Sequential) CloseActivations() {
	for _, t := range s.saved {
		t.Close()
	}
	s.saved = nil
	if s.output != nil {
		s.output.Close()
		s.output = nil
	}
}

// Close releases the parameters and gradients of every module.
func (s *Sequential) Close() {
	for _, m := range s.Modules {
		if closer, ok := m.(interface{ Close() }); ok {
			closer.Close()
		}
	}
}

// AdamW is the decoupled-weight-decay optimizer over module parameters.
type AdamW struct {
	params       []*tensor.Tensor
	grads        []*tensor.Tensor
	m            []*tensor.Tensor
	v            []*tensor.Tensor
	norm         *tensor.Tensor
	learningRate float64
	weightDecay  float64
	maxNorm      float64
	step         int
}

// NewAdamW creates optimizer state for the given parameters and gradients.
func NewAdamW(params, grads []*tensor.Tensor, learningRate, weightDecay, maxNorm float64) (*AdamW, error) {
	if len(params) != len(grads) {
		return nil, fmt.Errorf("nn: parameter and gradient counts differ")
	}
	device := tensor.CUDA
	if len(params) > 0 {
		device = params[0].Device()
	}
	norm, err := tensor.ZerosOn(device, 1)
	if err != nil {
		return nil, err
	}
	o := &AdamW{params: params, grads: grads, norm: norm,
		learningRate: learningRate, weightDecay: weightDecay, maxNorm: maxNorm}
	for _, p := range params {
		m, err := tensor.ZerosOn(device, p.Shape()...)
		if err != nil {
			o.Close()
			return nil, err
		}
		v, err := tensor.ZerosOn(device, p.Shape()...)
		if err != nil {
			o.Close()
			return nil, err
		}
		o.m = append(o.m, m)
		o.v = append(o.v, v)
	}
	return o, nil
}

// Step clips the global gradient norm and applies one AdamW update.
func (o *AdamW) Step() error {
	if o.norm.Device() == tensor.CPU {
		var sum float64
		for _, g := range o.grads {
			for _, v := range g.Data() {
				sum += float64(v) * float64(v)
			}
		}
		norm := math.Sqrt(sum)
		scale := 1.0
		if o.maxNorm > 0 && norm > o.maxNorm {
			scale = o.maxNorm / norm
		}
		o.step++
		bc1 := 1 - math.Pow(.9, float64(o.step))
		bc2 := 1 - math.Pow(.999, float64(o.step))
		for i, p := range o.params {
			for j := range p.Data() {
				g := float32(float64(o.grads[i].Data()[j]) * scale)
				m := .9*o.m[i].Data()[j] + .1*g
				v := .999*o.v[i].Data()[j] + .001*g*g
				o.m[i].Data()[j] = m
				o.v[i].Data()[j] = v
				p.Data()[j] = p.Data()[j]*(1-float32(o.learningRate*o.weightDecay)) - float32(o.learningRate)*float32(float64(m)/bc1)/(float32(math.Sqrt(float64(v)/bc2))+1e-8)
			}
		}
		return nil
	}
	if err := o.norm.Zero(); err != nil {
		return err
	}
	for _, g := range o.grads {
		if err := kernels.SumSquares(g.Buffer(), o.norm.Buffer(), g.Numel()); err != nil {
			return err
		}
	}
	if err := kernels.SqrtScalar(o.norm.Buffer()); err != nil {
		return err
	}
	o.step++
	biasCorrection1 := 1 - math.Pow(0.9, float64(o.step))
	biasCorrection2 := 1 - math.Pow(0.999, float64(o.step))
	stepSize := float32(o.learningRate / biasCorrection1)
	for i, p := range o.params {
		if err := kernels.AdamWUpdate(p.Buffer(), o.grads[i].Buffer(), o.m[i].Buffer(), o.v[i].Buffer(), o.norm.Buffer(),
			p.Numel(), 0.9, 0.999, stepSize, float32(biasCorrection2), float32(o.learningRate*o.weightDecay), 1e-8, float32(o.maxNorm)); err != nil {
			return err
		}
	}
	return nil
}

// Close releases the optimizer state.
func (o *AdamW) Close() {
	if o.norm != nil {
		o.norm.Close()
	}
	for _, t := range o.m {
		t.Close()
	}
	for _, t := range o.v {
		t.Close()
	}
}

// MSE returns the mean squared error and its gradient with respect to
// predicted.
func MSE(predicted, target []float32) (float64, []float32) {
	gradient := make([]float32, len(predicted))
	total := 0.0
	for i := range predicted {
		difference := float64(predicted[i] - target[i])
		total += difference * difference
		gradient[i] = float32(2 * difference / float64(len(predicted)))
	}
	return total / float64(len(predicted)), gradient
}
