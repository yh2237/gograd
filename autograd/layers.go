package autograd

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/yh2237/gograd/tensor"
)

// ParameterizedLayer exposes a module's named state for composition.
type ParameterizedLayer interface{ StateModule() *Module }

func layerParameter(shape []int, device tensor.Device, init func(int) float32) (*Tensor, error) {
	v := make([]float32, numel(shape))
	for i := range v {
		v[i] = init(i)
	}
	return New(v, shape, device, true)
}

// EmbeddingLayer accepts flattened indices with an explicit leading shape.
type EmbeddingLayer struct {
	Module     Module
	Weight     *Tensor
	PaddingIdx int
}

func NewEmbeddingLayer(count, width, paddingIdx int, device tensor.Device, rng *rand.Rand) (*EmbeddingLayer, error) {
	w, err := layerParameter([]int{count, width}, device, func(int) float32 { return float32(rng.NormFloat64()) })
	if err != nil {
		return nil, err
	}
	l := &EmbeddingLayer{Weight: w, PaddingIdx: paddingIdx}
	l.Module.Parameters = []Parameter{{"weight", w}}
	return l, nil
}
func (l *EmbeddingLayer) StateModule() *Module { return &l.Module }
func (l *EmbeddingLayer) Train(training bool)  { l.Module.Train(training) }
func (l *EmbeddingLayer) Forward(ids []int, shape []int) *Tensor {
	return EmbeddingWithPadding(l.Weight, ids, shape, l.PaddingIdx)
}

// Conv1dLayer uses batch-time-channel activations and PyTorch [out,in,k] weights.
type Conv1dLayer struct {
	Module       Module
	Weight, Bias *Tensor
	Dilation     int
}

func NewConv1dLayer(in, out, kernel, dilation int, device tensor.Device, rng *rand.Rand) (*Conv1dLayer, error) {
	if in < 1 || out < 1 || kernel < 1 || kernel%2 == 0 || dilation < 1 {
		return nil, fmt.Errorf("autograd: invalid Conv1d dimensions")
	}
	bound := float64(1) / math.Sqrt(float64(in*kernel))
	init := func(int) float32 { return float32((rng.Float64()*2 - 1) * bound) }
	w, e := layerParameter([]int{out, in, kernel}, device, init)
	if e != nil {
		return nil, e
	}
	b, e := layerParameter([]int{out}, device, init)
	if e != nil {
		w.Close()
		return nil, e
	}
	l := &Conv1dLayer{Weight: w, Bias: b, Dilation: dilation}
	l.Module.Parameters = []Parameter{{"weight", w}, {"bias", b}}
	return l, nil
}
func (l *Conv1dLayer) StateModule() *Module      { return &l.Module }
func (l *Conv1dLayer) Train(training bool)       { l.Module.Train(training) }
func (l *Conv1dLayer) Forward(x *Tensor) *Tensor { return Conv1dGEMM(x, l.Weight, l.Bias, l.Dilation) }

type LayerNormLayer struct {
	Module       Module
	Weight, Bias *Tensor
	Eps          float32
}

func NewLayerNormLayer(width int, eps float32, device tensor.Device) (*LayerNormLayer, error) {
	w, e := layerParameter([]int{width}, device, func(int) float32 { return 1 })
	if e != nil {
		return nil, e
	}
	b, e := Zeros([]int{width}, device, true)
	if e != nil {
		w.Close()
		return nil, e
	}
	l := &LayerNormLayer{Weight: w, Bias: b, Eps: eps}
	l.Module.Parameters = []Parameter{{"weight", w}, {"bias", b}}
	return l, nil
}
func (l *LayerNormLayer) StateModule() *Module      { return &l.Module }
func (l *LayerNormLayer) Train(training bool)       { l.Module.Train(training) }
func (l *LayerNormLayer) Forward(x *Tensor) *Tensor { return LayerNorm(x, l.Weight, l.Bias, l.Eps) }

// LinearLayer applies a PyTorch-layout [out,in] affine projection on the last axis.
type LinearLayer struct {
	Module       Module
	Weight, Bias *Tensor
}

func NewLinearLayer(in, out int, device tensor.Device, rng *rand.Rand) (*LinearLayer, error) {
	if in < 1 || out < 1 {
		return nil, fmt.Errorf("autograd: invalid Linear dimensions")
	}
	bound := 1 / math.Sqrt(float64(in))
	init := func(int) float32 { return float32((rng.Float64()*2 - 1) * bound) }
	w, e := layerParameter([]int{out, in}, device, init)
	if e != nil {
		return nil, e
	}
	b, e := layerParameter([]int{out}, device, init)
	if e != nil {
		w.Close()
		return nil, e
	}
	l := &LinearLayer{Weight: w, Bias: b}
	l.Module.Parameters = []Parameter{{"weight", w}, {"bias", b}}
	return l, nil
}
func (l *LinearLayer) StateModule() *Module { return &l.Module }
func (l *LinearLayer) Train(training bool)  { l.Module.Train(training) }
func (l *LinearLayer) Forward(x *Tensor) *Tensor {
	return Add(MatMul(x, Transpose(l.Weight, 0, 1)), l.Bias)
}

type DropoutLayer struct {
	Probability float32
	Training    bool
}

func (l *DropoutLayer) Forward(x *Tensor, seed uint32) *Tensor {
	return Dropout(x, l.Probability, seed, l.Training)
}

func (l *DropoutLayer) Train(training bool) { l.Training = training }

type GELULayer struct{ Approximate bool }

func (l GELULayer) Forward(x *Tensor) *Tensor { return GELU(x, l.Approximate) }

// Sequential applies tensor-to-tensor layers in order. Layers with additional
// inputs (embedding indices or a dropout seed) are composed in a plain struct.
type TensorLayer interface{ Forward(*Tensor) *Tensor }
type Sequential []TensorLayer

func (s Sequential) Train(training bool) {
	for _, layer := range s {
		if trainer, ok := layer.(Trainer); ok {
			trainer.Train(training)
		}
	}
}

func (s Sequential) Forward(x *Tensor) *Tensor {
	for _, l := range s {
		x = l.Forward(x)
	}
	return x
}
