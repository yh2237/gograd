package irodori

import (
	"fmt"
	"math"
)

// LowRankProjection holds the two linear maps used by one AdaLN branch.
// Weights use PyTorch's row-major [out,in] layout.
type LowRankProjection struct {
	Down []float32
	Up   []float32
	Bias []float32
}

// LowRankAdaLN mirrors the reference's shift, scale and gate projections.
type LowRankAdaLN struct {
	Width, Rank        int
	Epsilon            float32
	Shift, Scale, Gate LowRankProjection
}

func (p LowRankProjection) valid(width, rank int) bool {
	return len(p.Down) == rank*width && len(p.Up) == width*rank && len(p.Bias) == width
}

func (p LowRankProjection) apply(input []float32, width, rank int) []float32 {
	hidden := make([]float32, rank)
	for r := 0; r < rank; r++ {
		for j := 0; j < width; j++ {
			hidden[r] += p.Down[r*width+j] * silu(input[j])
		}
	}
	out := make([]float32, width)
	for j := 0; j < width; j++ {
		out[j] = input[j] + p.Bias[j]
		for r := 0; r < rank; r++ {
			out[j] += p.Up[j*rank+r] * hidden[r]
		}
	}
	return out
}

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

// Forward takes flattened [batch,width] x and [batch,3*width] conditioning.
// It returns the modulated activations and tanh residual gates.
func (a LowRankAdaLN) Forward(x, cond []float32) ([]float32, []float32, error) {
	w, r := a.Width, a.Rank
	if w <= 0 || r <= 0 || a.Epsilon <= 0 || len(x)%w != 0 || len(cond) != len(x)*3 || !a.Shift.valid(w, r) || !a.Scale.valid(w, r) || !a.Gate.valid(w, r) {
		return nil, nil, fmt.Errorf("irodori: invalid LowRankAdaLN shape")
	}
	out, gates := make([]float32, len(x)), make([]float32, len(x))
	for row := 0; row < len(x)/w; row++ {
		base := row * w
		c := cond[row*3*w : (row+1)*3*w]
		shift := a.Shift.apply(c[:w], w, r)
		scale := a.Scale.apply(c[w:2*w], w, r)
		gate := a.Gate.apply(c[2*w:], w, r)
		var sum float32
		for _, v := range x[base : base+w] {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum/float32(w)+a.Epsilon)))
		for j := 0; j < w; j++ {
			out[base+j] = x[base+j]*inv*(1+scale[j]) + shift[j]
			gates[base+j] = float32(math.Tanh(float64(gate[j])))
		}
	}
	return out, gates, nil
}
