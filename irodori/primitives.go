// Package irodori contains CPU inference primitives for Irodori-TTS.
// The full checkpoint graph and audio codec are not implemented yet.
package irodori

import (
	"fmt"
	"math"
)

// RMSNorm applies the reference model's last-axis normalization to contiguous
// rows. The caller owns the output; weights are shared across rows.
func RMSNorm(input, weight []float32, width int, eps float32) ([]float32, error) {
	if width <= 0 || len(weight) != width || len(input)%width != 0 || eps <= 0 {
		return nil, fmt.Errorf("irodori: invalid RMSNorm shape or epsilon")
	}
	out := make([]float32, len(input))
	for row := 0; row < len(input); row += width {
		var sum float32
		for _, v := range input[row : row+width] {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum/float32(width)+eps)))
		for j := range weight {
			out[row+j] = input[row+j] * inv * weight[j]
		}
	}
	return out, nil
}

// Rotary applies adjacent-pair RoPE to [batch, sequence, heads, headDim].
// The reference uses theta=10000 and float32 frequencies.
func Rotary(input []float32, batch, sequence, heads, headDim int) ([]float32, error) {
	if batch <= 0 || sequence <= 0 || heads <= 0 || headDim <= 0 || headDim%2 != 0 || len(input) != batch*sequence*heads*headDim {
		return nil, fmt.Errorf("irodori: invalid rotary shape")
	}
	out := make([]float32, len(input))
	for s := 0; s < sequence; s++ {
		for d := 0; d < headDim; d += 2 {
			freq := float32(1 / math.Pow(10000, float64(float32(d)/float32(headDim))))
			angle := float32(s) * freq
			cos, sin := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
			for b := 0; b < batch; b++ {
				for h := 0; h < heads; h++ {
					i := ((b*sequence+s)*heads+h)*headDim + d
					out[i] = input[i]*cos - input[i+1]*sin
					out[i+1] = input[i]*sin + input[i+1]*cos
				}
			}
		}
	}
	return out, nil
}

// TimestepEmbedding follows the DiT's cosine then sine embedding, including
// the reference's 1000 multiplier on its frequency vector.
func TimestepEmbedding(t []float32, dim int) ([]float32, error) {
	if dim <= 0 || dim%2 != 0 {
		return nil, fmt.Errorf("irodori: timestep dimension must be positive and even")
	}
	half := dim / 2
	out := make([]float32, len(t)*dim)
	logTheta := float32(math.Log(10000))
	for i, time := range t {
		for j := 0; j < half; j++ {
			freq := float32(1000) * float32(math.Exp(float64(-logTheta*float32(j)/float32(half))))
			arg := time * freq
			out[i*dim+j] = float32(math.Cos(float64(arg)))
			out[i*dim+half+j] = float32(math.Sin(float64(arg)))
		}
	}
	return out, nil
}

// RFSchedule returns the decreasing time grid used by the Euler RF sampler.
func RFSchedule(steps int, mode string, sway float64) ([]float32, error) {
	if steps <= 0 || math.IsNaN(sway) || math.IsInf(sway, 0) {
		return nil, fmt.Errorf("irodori: invalid RF schedule arguments")
	}
	if mode != "linear" && mode != "sway" {
		return nil, fmt.Errorf("irodori: unsupported RF schedule %q", mode)
	}
	out := make([]float32, steps+1)
	for i := range out {
		u := float32(i) / float32(steps)
		if mode == "sway" {
			u += float32(sway) * (float32(math.Cos(float64(float32(math.Pi/2)*u))) + u - 1)
			u = max(0, min(1, u))
		}
		out[i] = (1 - u) * 0.999
		if i > 0 && out[i] >= out[i-1] {
			return nil, fmt.Errorf("irodori: RF schedule is not strictly decreasing")
		}
	}
	return out, nil
}

// TemporalScoreRescale applies the optional sampling-time score correction.
func TemporalScoreRescale(velocity, latent []float32, t, k, sigma float32) ([]float32, error) {
	if len(velocity) != len(latent) || k <= 0 || t < 0 || t > 1 {
		return nil, fmt.Errorf("irodori: invalid temporal score inputs")
	}
	out := make([]float32, len(velocity))
	if t == 1 {
		copy(out, velocity)
		return out, nil
	}
	oneMinus := 1 - t
	snr := oneMinus * oneMinus / (t * t)
	sigmaSq := sigma * sigma
	ratio := (snr*sigmaSq + 1) / (snr*sigmaSq/k + 1)
	for i := range out {
		out[i] = (ratio*(oneMinus*velocity[i]+latent[i]) - latent[i]) / oneMinus
	}
	return out, nil
}
