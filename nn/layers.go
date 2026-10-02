package nn

import (
	"fmt"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"github.com/yh2237/gograd/tensor"
	"math"
	"math/rand"
)

func values(t *tensor.Tensor) ([]float32, error) {
	if t.Device() == tensor.CPU {
		return t.Data(), nil
	}
	return t.ToHost()
}
func output(device tensor.Device, shape []int, v []float32) (*tensor.Tensor, error) {
	return tensor.FromHostOn(device, shape, v)
}

// Conv1d consumes and returns [batch,time,channels]. Weight is
// [out_channels,in_channels,kernel] in cross-correlation order.
type Conv1d struct {
	Weight, Bias              *tensor.Tensor
	gradWeight, gradBias      *tensor.Tensor
	In, Out, Kernel, Dilation int
	input                     *tensor.Tensor
	columns                   []float32
	deviceColumns             *tensor.Tensor
}

func NewConv1d(in, out, kernel, dilation int, rng *rand.Rand) (*Conv1d, error) {
	return NewConv1dOn(tensor.CUDA, in, out, kernel, dilation, rng)
}
func NewConv1dOn(device tensor.Device, in, out, kernel, dilation int, rng *rand.Rand) (*Conv1d, error) {
	if in < 1 || out < 1 || kernel < 1 || kernel%2 == 0 || dilation < 1 {
		return nil, fmt.Errorf("nn: invalid Conv1d dimensions")
	}
	bound := 1 / math.Sqrt(float64(in*kernel))
	w := make([]float32, out*in*kernel)
	for i := range w {
		w[i] = float32((rng.Float64()*2 - 1) * bound)
	}
	weight, err := tensor.FromHostOn(device, []int{out, in, kernel}, w)
	if err != nil {
		return nil, err
	}
	bias, err := tensor.ZerosOn(device, out)
	if err != nil {
		weight.Close()
		return nil, err
	}
	gw, err := tensor.ZerosOn(device, out, in, kernel)
	if err != nil {
		weight.Close()
		bias.Close()
		return nil, err
	}
	gb, err := tensor.ZerosOn(device, out)
	if err != nil {
		weight.Close()
		bias.Close()
		gw.Close()
		return nil, err
	}
	return &Conv1d{Weight: weight, Bias: bias, gradWeight: gw, gradBias: gb, In: in, Out: out, Kernel: kernel, Dilation: dilation}, nil
}
func (c *Conv1d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{c.Weight, c.Bias} }
func (c *Conv1d) Gradients() []*tensor.Tensor  { return []*tensor.Tensor{c.gradWeight, c.gradBias} }
func (c *Conv1d) ZeroGrad()                    { _ = c.gradWeight.Zero(); _ = c.gradBias.Zero() }
func (c *Conv1d) Close() {
	for _, t := range []*tensor.Tensor{c.Weight, c.Bias, c.gradWeight, c.gradBias} {
		t.Close()
	}
}
func (c *Conv1d) Forward(blas *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	s := x.Shape()
	if len(s) != 3 || s[2] != c.In || x.Device() != c.Weight.Device() {
		return nil, fmt.Errorf("nn: Conv1d expects [batch,time,%d] on %s, got %v", c.In, c.Weight.Device(), s)
	}
	b, t := s[0], s[1]
	rows, k := b*t, c.In*c.Kernel
	if x.Device() == tensor.CUDA {
		return c.forwardCUDA(blas, x, b, t, rows, k)
	}
	xv, err := values(x)
	if err != nil {
		return nil, err
	}
	wv, err := values(c.Weight)
	if err != nil {
		return nil, err
	}
	bias, err := values(c.Bias)
	if err != nil {
		return nil, err
	}
	cols := make([]float32, rows*k)
	pad := c.Kernel / 2 * c.Dilation
	for n := 0; n < b; n++ {
		for at := 0; at < t; at++ {
			dst := cols[(n*t+at)*k:]
			for ci := 0; ci < c.In; ci++ {
				for j := 0; j < c.Kernel; j++ {
					src := at + j*c.Dilation - pad
					if src >= 0 && src < t {
						dst[ci*c.Kernel+j] = xv[(n*t+src)*c.In+ci]
					}
				}
			}
		}
	}
	wt := make([]float32, k*c.Out)
	for o := 0; o < c.Out; o++ {
		for j := 0; j < k; j++ {
			wt[j*c.Out+o] = wv[o*k+j]
		}
	}
	y := make([]float32, rows*c.Out)
	tensor.SGEMM(y, cols, wt, rows, c.Out, k)
	for r := 0; r < rows; r++ {
		for o := 0; o < c.Out; o++ {
			y[r*c.Out+o] += bias[o]
		}
	}
	c.input = x
	c.columns = cols
	return output(x.Device(), []int{b, t, c.Out}, y)
}
func (c *Conv1d) Backward(blas *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error) {
	if c.input == nil {
		return nil, fmt.Errorf("nn: Conv1d backward before forward")
	}
	s := c.input.Shape()
	b, t := s[0], s[1]
	rows, k := b*t, c.In*c.Kernel
	if gs := grad.Shape(); len(gs) != 3 || gs[0] != b || gs[1] != t || gs[2] != c.Out {
		return nil, fmt.Errorf("nn: Conv1d gradient shape mismatch")
	}
	if grad.Device() == tensor.CUDA {
		return c.backwardCUDA(blas, grad, s, b, t, rows, k)
	}
	gv, err := values(grad)
	if err != nil {
		return nil, err
	}
	wv, err := values(c.Weight)
	if err != nil {
		return nil, err
	}
	gw, err := values(c.gradWeight)
	if err != nil {
		return nil, err
	}
	gb, err := values(c.gradBias)
	if err != nil {
		return nil, err
	}
	// Transpose columns and multiply to form all parameter gradients.
	ct := make([]float32, k*rows)
	for r := 0; r < rows; r++ {
		for j := 0; j < k; j++ {
			ct[j*rows+r] = c.columns[r*k+j]
		}
	}
	dw := make([]float32, k*c.Out)
	tensor.SGEMM(dw, ct, gv, k, c.Out, rows)
	for o := 0; o < c.Out; o++ {
		for j := 0; j < k; j++ {
			gw[o*k+j] += dw[j*c.Out+o]
		}
	}
	for r := 0; r < rows; r++ {
		for o := 0; o < c.Out; o++ {
			gb[o] += gv[r*c.Out+o]
		}
	}
	dc := make([]float32, rows*k)
	tensor.SGEMM(dc, gv, wv, rows, k, c.Out)
	dx := make([]float32, b*t*c.In)
	pad := c.Kernel / 2 * c.Dilation
	for n := 0; n < b; n++ {
		for at := 0; at < t; at++ {
			src := dc[(n*t+at)*k:]
			for ci := 0; ci < c.In; ci++ {
				for j := 0; j < c.Kernel; j++ {
					dst := at + j*c.Dilation - pad
					if dst >= 0 && dst < t {
						dx[(n*t+dst)*c.In+ci] += src[ci*c.Kernel+j]
					}
				}
			}
		}
	}
	if c.Weight.Device() == tensor.CUDA {
		if err := c.gradWeight.CopyFrom(gw); err != nil {
			return nil, err
		}
		if err := c.gradBias.CopyFrom(gb); err != nil {
			return nil, err
		}
	}
	return output(grad.Device(), s, dx)
}

func (c *Conv1d) forwardCUDA(blas *cuda.Blas, x *tensor.Tensor, b, t, rows, k int) (*tensor.Tensor, error) {
	cols, err := tensor.NewOn(tensor.CUDA, rows, k)
	if err != nil {
		return nil, err
	}
	y, err := tensor.NewOn(tensor.CUDA, b, t, c.Out)
	if err != nil {
		cols.Close()
		return nil, err
	}
	if err = kernels.Im2col(x.Buffer(), cols.Buffer(), b, c.In, t, c.Kernel, c.Dilation); err == nil {
		err = blas.SgemmRowMajorNT(rows, c.Out, k, 1, cols.Buffer().Pointer(), k, c.Weight.Buffer().Pointer(), k, 0, y.Buffer().Pointer(), c.Out)
	}
	if err == nil {
		err = kernels.AddBiasColumns(y.Buffer(), c.Bias.Buffer(), rows, c.Out)
	}
	if err != nil {
		cols.Close()
		y.Close()
		return nil, err
	}
	if c.deviceColumns != nil {
		c.deviceColumns.Close()
	}
	c.deviceColumns, c.input = cols, x
	return y, nil
}

func (c *Conv1d) backwardCUDA(blas *cuda.Blas, grad *tensor.Tensor, shape []int, b, t, rows, k int) (*tensor.Tensor, error) {
	dx, err := tensor.ZerosOn(tensor.CUDA, shape...)
	if err != nil {
		return nil, err
	}
	dcols, err := tensor.NewOn(tensor.CUDA, rows, k)
	if err != nil {
		dx.Close()
		return nil, err
	}
	defer dcols.Close()
	err = blas.SgemmRowMajorTransposeA(c.Out, k, rows, 1, grad.Buffer().Pointer(), c.Out, c.deviceColumns.Buffer().Pointer(), k, 0, c.gradWeight.Buffer().Pointer(), k)
	if err == nil {
		err = kernels.ColumnSum(grad.Buffer(), c.gradBias.Buffer(), rows, c.Out)
	}
	if err == nil {
		err = blas.SgemmRowMajor(rows, k, c.Out, 1, grad.Buffer().Pointer(), c.Out, c.Weight.Buffer().Pointer(), k, 0, dcols.Buffer().Pointer(), k)
	}
	if err == nil {
		err = kernels.Col2im(dcols.Buffer(), dx.Buffer(), b, c.In, t, c.Kernel, c.Dilation)
	}
	if err != nil {
		dx.Close()
		return nil, err
	}
	c.deviceColumns.Close()
	c.deviceColumns = nil
	return dx, nil
}

// ReLU, LeakyReLU and GELU are elementwise trainable-graph modules.
type ReLU struct{ activation }
type LeakyReLU struct {
	activation
	Slope float32
}
type GELU struct{ activation }
type activation struct{ input *tensor.Tensor }

func (*ReLU) Parameters() []*tensor.Tensor      { return nil }
func (*ReLU) Gradients() []*tensor.Tensor       { return nil }
func (*ReLU) ZeroGrad()                         {}
func (*LeakyReLU) Parameters() []*tensor.Tensor { return nil }
func (*LeakyReLU) Gradients() []*tensor.Tensor  { return nil }
func (*LeakyReLU) ZeroGrad()                    {}
func (*GELU) Parameters() []*tensor.Tensor      { return nil }
func (*GELU) Gradients() []*tensor.Tensor       { return nil }
func (*GELU) ZeroGrad()                         {}
func gelu(x float32) float32 {
	v := float64(x)
	return float32(.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+.044715*v*v*v))))
}
func geluPrime(x float32) float32 {
	v := float64(x)
	u := math.Sqrt(2/math.Pi) * (v + .044715*v*v*v)
	th := math.Tanh(u)
	return float32(.5*(1+th) + .5*v*(1-th*th)*math.Sqrt(2/math.Pi)*(1+3*.044715*v*v))
}
func activationForward(a *activation, x *tensor.Tensor, f func(float32) float32) (*tensor.Tensor, error) {
	v, err := values(x)
	if err != nil {
		return nil, err
	}
	y := make([]float32, len(v))
	for i, z := range v {
		y[i] = f(z)
	}
	a.input = x
	return output(x.Device(), x.Shape(), y)
}
func activationBackward(a *activation, g *tensor.Tensor, f func(float32) float32) (*tensor.Tensor, error) {
	if a.input == nil {
		return nil, fmt.Errorf("nn: activation backward before forward")
	}
	v, err := values(a.input)
	if err != nil {
		return nil, err
	}
	gv, err := values(g)
	if err != nil {
		return nil, err
	}
	dx := make([]float32, len(v))
	for i, z := range v {
		dx[i] = gv[i] * f(z)
	}
	return output(g.Device(), g.Shape(), dx)
}
func (a *ReLU) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Device() == tensor.CUDA {
		return activationCUDA(&a.activation, x, nil, 0, 0)
	}
	return activationForward(&a.activation, x, func(v float32) float32 {
		if v > 0 {
			return v
		}
		return 0
	})
}
func (a *ReLU) Backward(_ *cuda.Blas, g *tensor.Tensor) (*tensor.Tensor, error) {
	if g.Device() == tensor.CUDA {
		return activationCUDA(&a.activation, a.input, g, 0, 0)
	}
	return activationBackward(&a.activation, g, func(v float32) float32 {
		if v > 0 {
			return 1
		}
		return 0
	})
}
func (a *LeakyReLU) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Device() == tensor.CUDA {
		return activationCUDA(&a.activation, x, nil, 1, a.Slope)
	}
	return activationForward(&a.activation, x, func(v float32) float32 {
		if v > 0 {
			return v
		}
		return a.Slope * v
	})
}
func (a *LeakyReLU) Backward(_ *cuda.Blas, g *tensor.Tensor) (*tensor.Tensor, error) {
	if g.Device() == tensor.CUDA {
		return activationCUDA(&a.activation, a.input, g, 1, a.Slope)
	}
	return activationBackward(&a.activation, g, func(v float32) float32 {
		if v > 0 {
			return 1
		}
		return a.Slope
	})
}
func (a *GELU) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Device() == tensor.CUDA {
		return activationCUDA(&a.activation, x, nil, 2, 0)
	}
	return activationForward(&a.activation, x, gelu)
}
func (a *GELU) Backward(_ *cuda.Blas, g *tensor.Tensor) (*tensor.Tensor, error) {
	if g.Device() == tensor.CUDA {
		return activationCUDA(&a.activation, a.input, g, 2, 0)
	}
	return activationBackward(&a.activation, g, geluPrime)
}

func activationCUDA(a *activation, x, grad *tensor.Tensor, kind int, slope float32) (*tensor.Tensor, error) {
	if x == nil {
		return nil, fmt.Errorf("nn: activation backward before forward")
	}
	y, err := tensor.NewOn(tensor.CUDA, x.Shape()...)
	if err != nil {
		return nil, err
	}
	var gb *cuda.Buffer
	if grad != nil {
		gb = grad.Buffer()
	} else {
		a.input = x
	}
	if err = kernels.Activation(x.Buffer(), gb, y.Buffer(), x.Numel(), kind, slope, grad != nil); err != nil {
		y.Close()
		return nil, err
	}
	return y, nil
}

// Residual computes x + F(x), requiring equal input and output shapes.
type Residual struct{ Inner Module }

func (r *Residual) Parameters() []*tensor.Tensor { return r.Inner.Parameters() }
func (r *Residual) Gradients() []*tensor.Tensor  { return r.Inner.Gradients() }
func (r *Residual) ZeroGrad()                    { r.Inner.ZeroGrad() }
func (r *Residual) Close() {
	if c, ok := r.Inner.(interface{ Close() }); ok {
		c.Close()
	}
}
func (r *Residual) Forward(blas *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := r.Inner.Forward(blas, x)
	if err != nil {
		return nil, err
	}
	if y.Numel() != x.Numel() {
		return nil, fmt.Errorf("nn: residual shape mismatch")
	}
	if x.Device() == tensor.CUDA {
		sum, err := tensor.NewOn(tensor.CUDA, x.Shape()...)
		if err != nil {
			return nil, err
		}
		if err = kernels.AddPair(sum.Buffer(), x.Buffer(), y.Buffer(), x.Numel()); err != nil {
			sum.Close()
			return nil, err
		}
		return sum, nil
	}
	xv, err := values(x)
	if err != nil {
		return nil, err
	}
	yv, err := values(y)
	if err != nil {
		return nil, err
	}
	sum := make([]float32, len(yv))
	for i := range sum {
		sum[i] = yv[i] + xv[i]
	}
	return output(x.Device(), x.Shape(), sum)
}
func (r *Residual) Backward(blas *cuda.Blas, g *tensor.Tensor) (*tensor.Tensor, error) {
	dx, err := r.Inner.Backward(blas, g)
	if err != nil {
		return nil, err
	}
	if dx.Device() == tensor.CUDA {
		return dx, kernels.AddInto(dx.Buffer(), g.Buffer(), dx.Numel())
	}
	dv, err := values(dx)
	if err != nil {
		return nil, err
	}
	gv, err := values(g)
	if err != nil {
		return nil, err
	}
	for i := range dv {
		dv[i] += gv[i]
	}
	if dx.Device() == tensor.CUDA {
		err = dx.CopyFrom(dv)
	}
	return dx, err
}
func (r *Residual) CloseActivations() {
	if c, ok := r.Inner.(interface{ CloseActivations() }); ok {
		c.CloseActivations()
	}
}
