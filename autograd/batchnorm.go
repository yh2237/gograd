package autograd

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/tensor"
)

// BatchNormOptions selects batch statistics in training or running statistics
// in evaluation. Channels are axis 1 by default, or the last axis when requested.
// Zero Eps defaults to 1e-5. Momentum is the new-statistic weight in [0,1]; zero
// freezes running values. Training without running buffers is supported.
type BatchNormOptions struct {
	Training, ChannelsLast bool
	Eps, Momentum          float32
}

func (o BatchNormOptions) normalized() (BatchNormOptions, error) {
	if o.Eps == 0 {
		o.Eps = 1e-5
	}
	if o.Eps <= 0 || math.IsNaN(float64(o.Eps)) || math.IsInf(float64(o.Eps), 0) || o.Momentum < 0 || o.Momentum > 1 || math.IsNaN(float64(o.Momentum)) {
		return o, fmt.Errorf("autograd: invalid BatchNorm epsilon or momentum")
	}
	return o, nil
}

// BatchNorm normalizes rank 2..6 float32 inputs per channel. Weight and bias
// are independently optional [channels] tensors. Running mean/variance must
// both be nil or non-gradient contiguous [channels] buffers; evaluation requires
// them. Training uses biased variance for output and unbiased variance for the
// running update. Running buffers are never differentiated.
func BatchNorm(x, weight, bias, runningMean, runningVar *Tensor, options BatchNormOptions) *Tensor {
	useCUDA := dispatchBackend("batch_norm", x.Device)
	return batchNorm(x, weight, bias, runningMean, runningVar, nil, options, false, useCUDA)
}

type batchNormSpec struct{ outer, channels, inner, samples int }

func batchNormShape(shape []int, channelsLast, training bool) (batchNormSpec, error) {
	s := batchNormSpec{}
	if len(shape) < 2 || len(shape) > 6 {
		return s, fmt.Errorf("autograd: BatchNorm requires rank 2..6")
	}
	n := 1
	for _, d := range shape {
		if d < 1 || d > 1<<31-1 || n > (1<<31-1)/d {
			return s, fmt.Errorf("autograd: invalid or oversized BatchNorm shape")
		}
		n *= d
	}
	axis := 1
	if channelsLast {
		axis = len(shape) - 1
	}
	s.outer, s.channels, s.inner = numel(shape[:axis]), shape[axis], numel(shape[axis+1:])
	s.samples = s.outer * s.inner
	if s.channels > (1<<31-1)/2 {
		return s, fmt.Errorf("autograd: oversized BatchNorm statistics")
	}
	if training && s.samples <= 1 {
		return s, fmt.Errorf("autograd: BatchNorm needs more than one sample per channel in training")
	}
	return s, nil
}

func batchNorm(x, weight, bias, mean, variance, count *Tensor, options BatchNormOptions, cumulative, useCUDA bool) *Tensor {
	same(x, weight, bias, mean, variance, count)
	o, err := options.normalized()
	if err != nil {
		panic(err)
	}
	s, err := batchNormShape(x.Shape, o.ChannelsLast, o.Training)
	if err != nil {
		panic(err)
	}
	if (mean == nil) != (variance == nil) || !o.Training && mean == nil {
		panic("autograd: BatchNorm requires paired running statistics for evaluation")
	}
	for _, v := range []*Tensor{weight, bias, mean, variance} {
		if v != nil && (len(v.Shape) != 1 || v.Shape[0] != s.channels) {
			panic("autograd: BatchNorm channel vector shape")
		}
	}
	for _, v := range []*Tensor{mean, variance, count} {
		if v != nil && (v.RequiresGrad || !v.IsContiguous()) {
			panic("autograd: BatchNorm state must be contiguous non-gradient buffers")
		}
	}
	if count != nil && (len(count.Shape) != 0 || mean == nil) {
		panic("autograd: BatchNorm counter requires scalar tracked state")
	}
	// Mutable state cannot alias another input/state allocation.
	all := []*Tensor{x, weight, bias, mean, variance, count}
	for i := 3; i < len(all); i++ {
		if all[i] == nil {
			continue
		}
		for j, v := range all {
			if j != i && v != nil && all[i].storage == v.storage {
				panic("autograd: BatchNorm running state aliases an input")
			}
		}
	}
	var context *ExecutionContext
	for _, v := range all {
		context = mergeExecutionContext(context, v)
	}
	x = x.Contiguous()
	if weight != nil {
		weight = weight.Contiguous()
	}
	if bias != nil {
		bias = bias.Contiguous()
	}
	parents := []*Tensor{x}
	if weight != nil {
		parents = append(parents, weight)
	}
	if bias != nil {
		parents = append(parents, bias)
	}
	if useCUDA {
		return gpuBatchNorm(x, weight, bias, mean, variance, count, o, s, cumulative, context, parents)
	}
	return cpuBatchNorm(x, weight, bias, mean, variance, count, o, s, cumulative, context, parents)
}

func (s batchNormSpec) index(sample, ch int) int {
	return (sample/s.inner*s.channels+ch)*s.inner + sample%s.inner
}

// Saved statistics are an independent ephemeral allocation/context dependency,
// rather than graph edges to mutable running buffers. This also propagates a
// context supplied only by those buffers to an otherwise-unbound input.
func cpuBatchNorm(x, w, b, mean, variance, count *Tensor, o BatchNormOptions, s batchNormSpec, cumulative bool, context *ExecutionContext, parents []*Tensor) *Tensor {
	stats, out := cpuAlloc(2*s.channels), cpuAlloc(x.Numel())
	momentum := o.Momentum
	if o.Training && count != nil {
		count.Data[0]++
		count.storage.version.Add(1)
		if cumulative {
			momentum = 1 / count.Data[0]
		}
	}
	for ch := 0; ch < s.channels; ch++ {
		mu, iv := float32(0), float32(0)
		if o.Training {
			var sum float64
			for sample := 0; sample < s.samples; sample++ {
				sum += float64(x.Data[s.index(sample, ch)])
			}
			avg := sum / float64(s.samples)
			var squares float64
			for sample := 0; sample < s.samples; sample++ {
				d := float64(x.Data[s.index(sample, ch)]) - avg
				squares += d * d
			}
			v := float32(squares / float64(s.samples))
			mu = float32(avg)
			iv = float32(1 / math.Sqrt(float64(v+o.Eps)))
			if mean != nil {
				mean.Data[ch] = (1-momentum)*mean.Data[ch] + momentum*mu
				unbiased := v * float32(s.samples) / float32(s.samples-1)
				variance.Data[ch] = (1-momentum)*variance.Data[ch] + momentum*unbiased
			}
		} else {
			mu = mean.Data[ch]
			iv = float32(1 / math.Sqrt(float64(variance.Data[ch]+o.Eps)))
		}
		stats[ch], stats[s.channels+ch] = mu, iv
	}
	if o.Training && mean != nil {
		mean.storage.version.Add(1)
		variance.storage.version.Add(1)
	}
	parallelFor(x.Numel(), func(start, end int) {
		for i := start; i < end; i++ {
			ch := i / s.inner % s.channels
			gamma, beta := float32(1), float32(0)
			if w != nil {
				gamma = w.Data[ch]
			}
			if b != nil {
				beta = b.Data[ch]
			}
			out[i] = (x.Data[i]-stats[ch])*stats[s.channels+ch]*gamma + beta
		}
	})
	saved := &Tensor{Data: stats, storage: cpuStorage(stats, true), Shape: []int{2, s.channels}, Strides: []int{s.channels, 1}, Device: tensor.CPU, DType: Float32, execution: context, ephemeral: true}
	return resultWithSaved(out, x.Shape, parents, []*Tensor{saved}, func(g []float32) {
		var dx, dw, db []float32
		if x.RequiresGrad {
			dx = cpuAlloc(x.Numel())
			defer cpuRelease(dx)
		}
		if w != nil && w.RequiresGrad {
			dw = cpuAlloc(s.channels)
			defer cpuRelease(dw)
		}
		if b != nil && b.RequiresGrad {
			db = cpuAlloc(s.channels)
			defer cpuRelease(db)
		}
		for ch := 0; ch < s.channels; ch++ {
			mu, iv := stats[ch], stats[s.channels+ch]
			var sum, sumNorm float64
			for sample := 0; sample < s.samples; sample++ {
				i := s.index(sample, ch)
				norm := (x.Data[i] - mu) * iv
				sum += float64(g[i])
				sumNorm += float64(g[i]) * float64(norm)
			}
			if dw != nil {
				dw[ch] = float32(sumNorm)
			}
			if db != nil {
				db[ch] = float32(sum)
			}
			if dx != nil {
				gamma := float32(1)
				if w != nil {
					gamma = w.Data[ch]
				}
				avg, avgNorm := float32(sum/float64(s.samples)), float32(sumNorm/float64(s.samples))
				for sample := 0; sample < s.samples; sample++ {
					i := s.index(sample, ch)
					value := g[i]
					if o.Training {
						value -= avg + (x.Data[i]-mu)*iv*avgNorm
					}
					dx[i] = value * gamma * iv
				}
			}
		}
		if dx != nil {
			x.addGrad(dx)
		}
		if dw != nil {
			w.addGrad(dw)
		}
		if db != nil {
			b.addGrad(db)
		}
	})
}
