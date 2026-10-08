package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
)

// ReferenceEncoder maps the codec's 32-wide latent frames to the speaker
// context used by the v4.1 denoiser and duration predictor.
type ReferenceEncoder struct{ State *autograd.SafeTensorFile }

func (e ReferenceEncoder) parameter(name string, shape ...int) ([]float32, error) {
	v, s, err := e.State.ReadF32(name)
	if err != nil {
		return nil, err
	}
	if len(s) != len(shape) {
		return nil, fmt.Errorf("irodori: %s shape %v", name, s)
	}
	for i := range shape {
		if s[i] != shape[i] {
			return nil, fmt.Errorf("irodori: %s shape %v", name, s)
		}
	}
	return v, nil
}

func (e ReferenceEncoder) project(name string, x []float32, rows, in, out int, bias bool) ([]float32, error) {
	w, err := e.parameter(name+".weight", out, in)
	if err != nil {
		return nil, err
	}
	y := linearRows(x, w, rows, in, out)
	if bias {
		b, err := e.parameter(name+".bias", out)
		if err != nil {
			return nil, err
		}
		for i := 0; i < rows; i++ {
			for j := 0; j < out; j++ {
				y[i*out+j] += b[j]
			}
		}
	}
	return y, nil
}

func (e ReferenceEncoder) attention(base string, x []float32, rows int) ([]float32, error) {
	const width, heads, headDim = 768, 12, 64
	q, err := e.project(base+"wq", x, rows, width, width, false)
	if err != nil {
		return nil, err
	}
	k, err := e.project(base+"wk", x, rows, width, width, false)
	if err != nil {
		return nil, err
	}
	v, err := e.project(base+"wv", x, rows, width, width, false)
	if err != nil {
		return nil, err
	}
	gate, err := e.project(base+"gate", x, rows, width, width, false)
	if err != nil {
		return nil, err
	}
	qw, err := e.parameter(base+"q_norm.weight", heads, headDim)
	if err != nil {
		return nil, err
	}
	kw, err := e.parameter(base+"k_norm.weight", heads, headDim)
	if err != nil {
		return nil, err
	}
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			start := row*width + head*headDim
			var qs, ks float32
			for d := 0; d < headDim; d++ {
				qs += q[start+d] * q[start+d]
				ks += k[start+d] * k[start+d]
			}
			qi := float32(1 / math.Sqrt(float64(qs/headDim+1e-5)))
			ki := float32(1 / math.Sqrt(float64(ks/headDim+1e-5)))
			for d := 0; d < headDim; d++ {
				q[start+d] *= qi * qw[head*headDim+d]
				k[start+d] *= ki * kw[head*headDim+d]
			}
			for d := 0; d < headDim; d += 2 {
				angle := float32(row) * float32(1/math.Pow(10000, float64(float32(d)/headDim)))
				co, si := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
				a, b := q[start+d], q[start+d+1]
				q[start+d], q[start+d+1] = a*co-b*si, a*si+b*co
				a, b = k[start+d], k[start+d+1]
				k[start+d], k[start+d+1] = a*co-b*si, a*si+b*co
			}
		}
	}
	context := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			scores := make([]float32, rows)
			maxScore := float32(math.Inf(-1))
			for key := 0; key < rows; key++ {
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[row*width+head*headDim+d] * k[key*width+head*headDim+d]
				}
				scores[key] = dot * 0.125
				maxScore = max(maxScore, scores[key])
			}
			var sum float32
			for key := range scores {
				scores[key] = float32(math.Exp(float64(scores[key] - maxScore)))
				sum += scores[key]
			}
			for key, score := range scores {
				p := score / sum
				for d := 0; d < headDim; d++ {
					context[row*width+head*headDim+d] += p * v[key*width+head*headDim+d]
				}
			}
		}
	}
	for i := range context {
		context[i] *= 1 / (1 + float32(math.Exp(float64(-gate[i]))))
	}
	return e.project(base+"wo", context, rows, width, width, false)
}

func (e ReferenceEncoder) mlp(base string, x []float32, rows int) ([]float32, error) {
	const width, hidden = 768, 1996 // int(768 * 2.6)
	a, err := e.project(base+"w1", x, rows, width, hidden, false)
	if err != nil {
		return nil, err
	}
	b, err := e.project(base+"w3", x, rows, width, hidden, false)
	if err != nil {
		return nil, err
	}
	for i := range a {
		a[i] = silu(a[i]) * b[i]
	}
	return e.project(base+"w2", a, rows, hidden, width, false)
}

// Forward accepts time-major codec latents and returns [1+floor(T/4),768].
// The first row is the masked mean of the remaining speaker tokens.
func (e ReferenceEncoder) Forward(latent []float32) ([]float32, error) {
	if e.State == nil || len(latent) == 0 || len(latent)%32 != 0 {
		return nil, fmt.Errorf("irodori: invalid reference latent")
	}
	const width, patch = 768, 4
	frames := len(latent) / 32
	rows := frames / patch
	if rows == 0 {
		return nil, fmt.Errorf("irodori: reference shorter than one patch")
	}
	packed := make([]float32, rows*128)
	copy(packed, latent)
	x, err := e.project("speaker_encoder.in_proj", packed, rows, 128, width, true)
	if err != nil {
		return nil, err
	}
	for i := range x {
		x[i] /= 6
	}
	for layer := 0; layer < 8; layer++ {
		base := fmt.Sprintf("speaker_encoder.blocks.%d.", layer)
		w, err := e.parameter(base+"attention_norm.weight", width)
		if err != nil {
			return nil, err
		}
		h, err := RMSNorm(x, w, width, 1e-5)
		if err != nil {
			return nil, err
		}
		a, err := e.attention(base+"attention.", h, rows)
		if err != nil {
			return nil, err
		}
		for i := range x {
			x[i] += a[i]
		}
		w, err = e.parameter(base+"mlp_norm.weight", width)
		if err != nil {
			return nil, err
		}
		h, err = RMSNorm(x, w, width, 1e-5)
		if err != nil {
			return nil, err
		}
		a, err = e.mlp(base+"mlp.", h, rows)
		if err != nil {
			return nil, err
		}
		for i := range x {
			x[i] += a[i]
		}
	}
	w, err := e.parameter("speaker_norm.weight", width)
	if err != nil {
		return nil, err
	}
	x, err = RMSNorm(x, w, width, 1e-5)
	if err != nil {
		return nil, err
	}
	out := make([]float32, (rows+1)*width)
	copy(out[width:], x)
	for row := 0; row < rows; row++ {
		for j := 0; j < width; j++ {
			out[j] += x[row*width+j]
		}
	}
	for j := 0; j < width; j++ {
		out[j] /= float32(rows)
	}
	return out, nil
}
