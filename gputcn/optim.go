package gputcn

import (
	"math"

	"github.com/yh2237/gograd/cuda"
)

// Parameters lists the device parameters in a stable order.
func (m *Model) Parameters() []*cuda.Buffer {
	params := []*cuda.Buffer{m.InputWeight, m.InputBias}
	for i := range m.Layers {
		params = append(params, m.Layers[i].Weight, m.Layers[i].Bias)
	}
	return append(params, m.OutputWeight, m.OutputBias)
}

func (g *Gradients) gradientBuffers() []*cuda.Buffer {
	grads := []*cuda.Buffer{g.InputWeight, g.InputBias}
	for i := range g.Layers {
		grads = append(grads, g.Layers[i].Weight, g.Layers[i].Bias)
	}
	return append(grads, g.OutputWeight, g.OutputBias)
}

// AdamState holds the device moment estimates and the global norm scratch.
type AdamState struct {
	step int
	m    []*cuda.Buffer
	v    []*cuda.Buffer
	norm *cuda.Buffer
}

// NewAdamState returns empty optimizer state.
func NewAdamState() *AdamState { return &AdamState{} }

// Close releases the moment buffers.
func (s *AdamState) Close() {
	for _, buffer := range s.m {
		buffer.Free()
	}
	for _, buffer := range s.v {
		buffer.Free()
	}
	if s.norm != nil {
		s.norm.Free()
	}
	s.m, s.v, s.norm = nil, nil, nil
}

func (s *AdamState) ensure(params []*cuda.Buffer) error {
	if len(s.m) == len(params) {
		return nil
	}
	s.Close()
	norm, err := cuda.Alloc(4)
	if err != nil {
		return err
	}
	if err := norm.Memset(0, 4); err != nil {
		return err
	}
	s.norm = norm
	for _, param := range params {
		m, err := cuda.Alloc(param.Size())
		if err != nil {
			return err
		}
		if err := m.Memset(0, m.Size()); err != nil {
			return err
		}
		v, err := cuda.Alloc(param.Size())
		if err != nil {
			return err
		}
		if err := v.Memset(0, v.Size()); err != nil {
			return err
		}
		s.m = append(s.m, m)
		s.v = append(s.v, v)
	}
	return nil
}

// ApplyAdamW clips the global gradient norm to maxNorm and applies one AdamW
// update on the device. The moment buffers live in state and are created on
// the first call.
func (m *Model) ApplyAdamW(grads *Gradients, state *AdamState, learningRate, weightDecay, maxNorm float64) error {
	params := m.Parameters()
	gradientBuffers := grads.gradientBuffers()
	if len(params) != len(gradientBuffers) {
		return errParameterMismatch
	}
	if err := state.ensure(params); err != nil {
		return err
	}
	if err := state.norm.Memset(0, 4); err != nil {
		return err
	}
	for _, gradient := range gradientBuffers {
		if err := cuda.SumSquares(gradient, state.norm, gradient.Size()/4); err != nil {
			return err
		}
	}
	if err := cuda.SqrtScalar(state.norm); err != nil {
		return err
	}
	state.step++
	biasCorrection1 := 1 - math.Pow(0.9, float64(state.step))
	biasCorrection2 := 1 - math.Pow(0.999, float64(state.step))
	stepSize := float32(learningRate / biasCorrection1)
	for i, param := range params {
		if err := cuda.AdamWUpdate(param, gradientBuffers[i], state.m[i], state.v[i], state.norm, param.Size()/4,
			0.9, 0.999, stepSize, float32(biasCorrection2), float32(learningRate*weightDecay), 1e-8, float32(maxNorm)); err != nil {
			return err
		}
	}
	return cuda.Synchronize()
}

var errParameterMismatch = &cuda.Error{Op: "ApplyAdamW", Code: -1, Message: "parameter and gradient counts differ"}
