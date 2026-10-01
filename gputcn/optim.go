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

// AdamState stores the moment estimates for ApplyAdamW.
type AdamState struct {
	step int
	m    [][]float64
	v    [][]float64
}

// NewAdamState returns empty optimizer state.
func NewAdamState() *AdamState { return &AdamState{} }

// ApplyAdamW clips the gradient norm to maxNorm and applies one AdamW update,
// matching the CPU optimizer's order: global norm clip, decoupled weight decay,
// then the adaptive step.
func (m *Model) ApplyAdamW(grads *Gradients, state *AdamState, learningRate, weightDecay, maxNorm float64) error {
	params := m.Parameters()
	gradientBuffers := grads.gradientBuffers()
	if len(params) != len(gradientBuffers) {
		return errParameterMismatch
	}
	gradValues := make([][]float32, len(gradientBuffers))
	total := 0.0
	for i, buffer := range gradientBuffers {
		values, err := DownloadFloat32(buffer, buffer.Size()/4)
		if err != nil {
			return err
		}
		gradValues[i] = values
		for _, value := range values {
			total += float64(value) * float64(value)
		}
	}
	total = math.Sqrt(total)
	coefficient := maxNorm / (total + 1e-6)
	if coefficient > 1 {
		coefficient = 1
	}
	if state.m == nil {
		state.m = make([][]float64, len(params))
		state.v = make([][]float64, len(params))
		for i, param := range params {
			state.m[i] = make([]float64, param.Size()/4)
			state.v[i] = make([]float64, param.Size()/4)
		}
	}
	state.step++
	biasCorrection1 := 1 - math.Pow(0.9, float64(state.step))
	biasCorrection2 := 1 - math.Pow(0.999, float64(state.step))
	stepSize := learningRate / biasCorrection1
	for i, param := range params {
		values, err := DownloadFloat32(param, param.Size()/4)
		if err != nil {
			return err
		}
		for j := range values {
			g := float64(gradValues[i][j]) * coefficient
			m := 0.9*state.m[i][j] + 0.1*g
			v := 0.999*state.v[i][j] + 0.001*g*g
			state.m[i][j] = m
			state.v[i][j] = v
			denom := math.Sqrt(v)/math.Sqrt(biasCorrection2) + 1e-8
			value := float64(values[j]) * (1 - learningRate*weightDecay)
			value -= stepSize * m / denom
			values[j] = float32(value)
		}
		if err := param.CopyFromHost(floatsToBytes(values)); err != nil {
			return err
		}
	}
	return nil
}

var errParameterMismatch = &cuda.Error{Op: "ApplyAdamW", Code: -1, Message: "parameter and gradient counts differ"}
