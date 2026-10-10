package autograd

import (
	"fmt"

	"github.com/yh2237/gograd/tensor"
)

// BatchNormLayerOptions defaults to affine, tracked statistics, eps=1e-5 and
// momentum=0.1. A non-nil Momentum overrides the default, including zero.
// CumulativeMomentum averages observed batch means/unbiased variances instead.
type BatchNormLayerOptions struct {
	Eps                                                                  float32
	Momentum                                                             *float32
	DisableAffine, DisableRunningStats, ChannelsLast, CumulativeMomentum bool
}

type BatchNormLayer struct {
	Module                                                   Module
	Weight, Bias, RunningMean, RunningVar, NumBatchesTracked *Tensor
	Eps, Momentum                                            float32
	Training, ChannelsLast, CumulativeMomentum               bool
}

func NewBatchNormLayer(channels int, options BatchNormLayerOptions, device tensor.Device) (*BatchNormLayer, error) {
	momentum := float32(.1)
	if options.Momentum != nil {
		momentum = *options.Momentum
	}
	o, err := (BatchNormOptions{Eps: options.Eps, Momentum: momentum}).normalized()
	if err != nil {
		return nil, err
	}
	if channels < 1 || channels > (1<<31-1)/2 {
		return nil, fmt.Errorf("autograd: invalid BatchNormLayer channel count")
	}
	l := &BatchNormLayer{Eps: o.Eps, Momentum: o.Momentum, ChannelsLast: options.ChannelsLast, CumulativeMomentum: options.CumulativeMomentum}
	newVector := func(name string, value float32, grad bool, buffer bool) (*Tensor, error) {
		v, err := layerParameter([]int{channels}, device, func(int) float32 { return value })
		if err != nil {
			return nil, err
		}
		v.RequiresGrad = grad
		if buffer {
			l.Module.Buffers = append(l.Module.Buffers, Parameter{name, v})
		} else {
			l.Module.Parameters = append(l.Module.Parameters, Parameter{name, v})
		}
		return v, nil
	}
	if !options.DisableAffine {
		l.Weight, err = newVector("weight", 1, true, false)
		if err == nil {
			l.Bias, err = newVector("bias", 0, true, false)
		}
		if err != nil {
			l.Close()
			return nil, err
		}
	}
	if !options.DisableRunningStats {
		l.RunningMean, err = newVector("running_mean", 0, false, true)
		if err == nil {
			l.RunningVar, err = newVector("running_var", 1, false, true)
		}
		if err == nil {
			l.NumBatchesTracked, err = Zeros([]int{}, device, false)
		}
		if err != nil {
			l.Close()
			return nil, err
		}
		l.Module.Buffers = append(l.Module.Buffers, Parameter{"num_batches_tracked", l.NumBatchesTracked})
	}
	l.Module.OnTrainingChange = func(training bool) { l.Training = training }
	l.Train(true)
	return l, nil
}

func (l *BatchNormLayer) StateModule() *Module { return &l.Module }
func (l *BatchNormLayer) Train(training bool)  { l.Module.Train(training) }
func (l *BatchNormLayer) Close() {
	for _, entries := range [][]Parameter{l.Module.Parameters, l.Module.Buffers} {
		for _, p := range entries {
			p.Value.Close()
		}
	}
}
func (l *BatchNormLayer) Forward(x *Tensor) *Tensor {
	useCUDA := dispatchBackend("batch_norm", x.Device)
	o := BatchNormOptions{Training: l.Training || l.RunningMean == nil, ChannelsLast: l.ChannelsLast, Eps: l.Eps, Momentum: l.Momentum}
	return batchNorm(x, l.Weight, l.Bias, l.RunningMean, l.RunningVar, l.NumBatchesTracked, o, l.CumulativeMomentum, useCUDA)
}
