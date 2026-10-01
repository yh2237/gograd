package gograd

import (
	"math"
	"math/rand"
)

// TCNLayer is one residual dilated convolution block.
type TCNLayer struct {
	Weight   *Tensor // [H,H,K]
	Bias     *Tensor // [H]
	Dilation int
}

// FrameIntonationTCN is a frame-level residual dilated TCN that maps [B,T,I]
// features to [B,T] intonation cents.
type FrameIntonationTCN struct {
	InputWeight  *Tensor // [H,I]
	InputBias    *Tensor // [H]
	Layers       []TCNLayer
	OutputWeight *Tensor // [1,H]
	OutputBias   *Tensor // [1]
}

// NewFrameIntonationTCN builds a model with all parameters set to zero.
func NewFrameIntonationTCN(inputs, hidden int, dilations []int) *FrameIntonationTCN {
	model := &FrameIntonationTCN{
		InputWeight:  Zeros(hidden, inputs),
		InputBias:    Zeros(hidden),
		OutputWeight: Zeros(1, hidden),
		OutputBias:   Zeros(1),
	}
	for _, dilation := range dilations {
		model.Layers = append(model.Layers, TCNLayer{
			Weight:   Zeros(hidden, hidden, 3),
			Bias:     Zeros(hidden),
			Dilation: dilation,
		})
	}
	return model
}

// Parameters returns every trainable tensor in a stable order.
func (m *FrameIntonationTCN) Parameters() []*Tensor {
	params := []*Tensor{m.InputWeight, m.InputBias}
	for i := range m.Layers {
		params = append(params, m.Layers[i].Weight, m.Layers[i].Bias)
	}
	return append(params, m.OutputWeight, m.OutputBias)
}

// ZeroGrad clears every parameter gradient.
func (m *FrameIntonationTCN) ZeroGrad() {
	for _, param := range m.Parameters() {
		param.ZeroGrad()
	}
}

// Forward applies the TCN and returns [B,T].
func (m *FrameIntonationTCN) Forward(values *Tensor) *Tensor {
	state := Tanh(Linear(values, m.InputWeight, m.InputBias)) // [B,T,H]
	state = Transpose12(state)                                // [B,H,T]
	for i := range m.Layers {
		layer := &m.Layers[i]
		convolved := Conv1d(state, layer.Weight, layer.Bias, layer.Dilation)
		state = Tanh(Add(state, convolved))
	}
	state = Transpose12(state) // [B,T,H]
	return SqueezeLast(Linear(state, m.OutputWeight, m.OutputBias))
}

// InitializeParameters fills the parameters with the same uniform scheme as
// PyTorch's default Linear and Conv1d initialization, using a local RNG so
// that runs are reproducible.
func (m *FrameIntonationTCN) InitializeParameters(seed int64) {
	rng := rand.New(rand.NewSource(seed))
	uniform := func(size, fanIn int) []float64 {
		bound := 1.0 / math.Sqrt(float64(fanIn))
		values := make([]float64, size)
		for i := range values {
			values[i] = (rng.Float64()*2 - 1) * bound
		}
		return values
	}
	copy(m.InputWeight.Data, uniform(m.InputWeight.Numel(), m.InputWeight.Shape[1]))
	copy(m.InputBias.Data, uniform(m.InputBias.Numel(), m.InputWeight.Shape[1]))
	for i := range m.Layers {
		layer := &m.Layers[i]
		fanIn := layer.Weight.Shape[1] * layer.Weight.Shape[2]
		copy(layer.Weight.Data, uniform(layer.Weight.Numel(), fanIn))
		copy(layer.Bias.Data, uniform(layer.Bias.Numel(), fanIn))
	}
	copy(m.OutputWeight.Data, uniform(m.OutputWeight.Numel(), m.OutputWeight.Shape[1]))
	copy(m.OutputBias.Data, uniform(m.OutputBias.Numel(), m.OutputWeight.Shape[1]))
}
