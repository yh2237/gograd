package autograd

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/yh2237/gograd/tensor"
)

// Conv2dLayer is an NCHW convolution with PyTorch-layout weights and named
// weight/bias state. Bias is nil when disabled at construction.
type Conv2dLayer struct {
	Module       Module
	Weight, Bias *Tensor
	Options      Conv2dOptions
}

// NewConv2dLayer uses PyTorch's default uniform initialization bound
// 1/sqrt((in/groups)*kH*kW). Kernel is [height,width], and bias controls whether
// a bias parameter is registered. The caller supplies the initialization RNG.
func NewConv2dLayer(in, out int, kernel [2]int, options Conv2dOptions, bias bool, device tensor.Device, rng *rand.Rand) (*Conv2dLayer, error) {
	o, err := options.normalized()
	if err != nil {
		return nil, err
	}
	if in < 1 || out < 1 || in > 1<<31-1 || out > 1<<31-1 || in%o.Groups != 0 || out%o.Groups != 0 || rng == nil {
		return nil, fmt.Errorf("autograd: invalid Conv2dLayer channels, groups or RNG")
	}
	shape := []int{out, in / o.Groups, kernel[0], kernel[1]}
	if _, err = conv2dProduct(shape...); err != nil {
		return nil, err
	}
	bound := 1 / math.Sqrt(float64((in/o.Groups)*kernel[0]*kernel[1]))
	init := func(int) float32 { return float32((rng.Float64()*2 - 1) * bound) }
	w, err := layerParameter(shape, device, init)
	if err != nil {
		return nil, err
	}
	l := &Conv2dLayer{Weight: w, Options: o}
	l.Module.Parameters = []Parameter{{"weight", w}}
	if bias {
		l.Bias, err = layerParameter([]int{out}, device, init)
		if err != nil {
			w.Close()
			return nil, err
		}
		l.Module.Parameters = append(l.Module.Parameters, Parameter{"bias", l.Bias})
	}
	return l, nil
}

func (l *Conv2dLayer) StateModule() *Module { return &l.Module }
func (l *Conv2dLayer) Train(training bool)  { l.Module.Train(training) }
func (l *Conv2dLayer) Forward(x *Tensor) *Tensor {
	return Conv2d(x, l.Weight, l.Bias, l.Options)
}
