package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
)

// CheckpointDiT runs a v4.1-Small denoiser forward with already-encoded
// text, speaker and caption contexts. Each block's weights are streamed.
type CheckpointDiT struct{ State *autograd.SafeTensorFile }

func (p CheckpointDiT) parameter(name string, shape ...int) ([]float32, error) {
	v, s, err := p.State.ReadF32(name)
	if err != nil {
		return nil, err
	}
	if len(s) != len(shape) {
		return nil, fmt.Errorf("irodori: DiT %s shape %v", name, s)
	}
	for i, d := range shape {
		if s[i] != d {
			return nil, fmt.Errorf("irodori: DiT %s shape %v", name, s)
		}
	}
	return v, nil
}

func (p CheckpointDiT) project(name string, x []float32, rows, in, out int, bias bool) ([]float32, error) {
	w, err := p.parameter(name+".weight", out, in)
	if err != nil {
		return nil, err
	}
	y := linearRows(x, w, rows, in, out)
	if bias {
		b, err := p.parameter(name+".bias", out)
		if err != nil {
			return nil, err
		}
		for row := 0; row < rows; row++ {
			for j := 0; j < out; j++ {
				y[row*out+j] += b[j]
			}
		}
	}
	return y, nil
}

func (p CheckpointDiT) adaln(name string, x, cond []float32) ([]float32, []float32, error) {
	branch := func(part string) (LowRankProjection, error) {
		down, err := p.parameter(name+part+"_down.weight", 192, 1280)
		if err != nil {
			return LowRankProjection{}, err
		}
		up, err := p.parameter(name+part+"_up.weight", 1280, 192)
		if err != nil {
			return LowRankProjection{}, err
		}
		bias, err := p.parameter(name+part+"_up.bias", 1280)
		if err != nil {
			return LowRankProjection{}, err
		}
		return LowRankProjection{Down: down, Up: up, Bias: bias}, nil
	}
	shift, err := branch("shift")
	if err != nil {
		return nil, nil, err
	}
	scale, err := branch("scale")
	if err != nil {
		return nil, nil, err
	}
	gate, err := branch("gate")
	if err != nil {
		return nil, nil, err
	}
	repeated := make([]float32, len(x)*3)
	for row := 0; row < len(x)/1280; row++ {
		copy(repeated[row*3840:(row+1)*3840], cond)
	}
	return (LowRankAdaLN{Width: 1280, Rank: 192, Epsilon: 1e-5, Shift: shift, Scale: scale, Gate: gate}).Forward(x, repeated)
}

func normalizeHeads(x, w []float32) {
	for start := 0; start < len(x); start += 64 {
		var sum float32
		for _, v := range x[start : start+64] {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum/64+1e-5)))
		for d := 0; d < 64; d++ {
			x[start+d] *= inv * w[(start/64%20)*64+d]
		}
	}
}

func (p CheckpointDiT) attention(name string, x, text, speaker, caption []float32, textMask, speakerMask, captionMask []bool) ([]float32, error) {
	const dim, heads, headDim = 1280, 20, 64
	seq, textLen, speakerLen, captionLen := len(x)/dim, len(textMask), len(speakerMask), len(captionMask)
	q, err := p.project(name+"wq", x, seq, dim, dim, false)
	if err != nil {
		return nil, err
	}
	ks, err := p.project(name+"wk", x, seq, dim, dim, false)
	if err != nil {
		return nil, err
	}
	vs, err := p.project(name+"wv", x, seq, dim, dim, false)
	if err != nil {
		return nil, err
	}
	kt, err := p.project(name+"wk_text", text, textLen, 512, dim, false)
	if err != nil {
		return nil, err
	}
	vt, err := p.project(name+"wv_text", text, textLen, 512, dim, false)
	if err != nil {
		return nil, err
	}
	ksp, err := p.project(name+"wk_speaker", speaker, speakerLen, 768, dim, false)
	if err != nil {
		return nil, err
	}
	vsp, err := p.project(name+"wv_speaker", speaker, speakerLen, 768, dim, false)
	if err != nil {
		return nil, err
	}
	kc, err := p.project(name+"wk_caption", caption, captionLen, 512, dim, false)
	if err != nil {
		return nil, err
	}
	vc, err := p.project(name+"wv_caption", caption, captionLen, 512, dim, false)
	if err != nil {
		return nil, err
	}
	qweight, err := p.parameter(name+"q_norm.weight", heads, headDim)
	if err != nil {
		return nil, err
	}
	kweight, err := p.parameter(name+"k_norm.weight", heads, headDim)
	if err != nil {
		return nil, err
	}
	normalizeHeads(q, qweight)
	normalizeHeads(ks, kweight)
	normalizeHeads(kt, kweight)
	normalizeHeads(ksp, kweight)
	normalizeHeads(kc, kweight)
	for pos := 0; pos < seq; pos++ {
		for h := 0; h < heads/2; h++ {
			for d := 0; d < headDim; d += 2 {
				freq := float32(1 / math.Pow(10000, float64(float32(d)/headDim)))
				angle := float32(pos) * freq
				co, si := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
				i := pos*dim + h*headDim + d
				qa, qb, ka, kb := q[i], q[i+1], ks[i], ks[i+1]
				q[i], q[i+1] = qa*co-qb*si, qa*si+qb*co
				ks[i], ks[i+1] = ka*co-kb*si, ka*si+kb*co
			}
		}
	}
	allK := append(append(append(ks, kt...), ksp...), kc...)
	allV := append(append(append(vs, vt...), vsp...), vc...)
	mask := make([]bool, seq+textLen+speakerLen+captionLen)
	for i := 0; i < seq; i++ {
		mask[i] = true
	}
	copy(mask[seq:], textMask)
	copy(mask[seq+textLen:], speakerMask)
	copy(mask[seq+textLen+speakerLen:], captionMask)
	context := make([]float32, seq*dim)
	for row := 0; row < seq; row++ {
		for h := 0; h < heads; h++ {
			scores := make([]float32, len(mask))
			maxScore := float32(math.Inf(-1))
			for key, valid := range mask {
				if !valid {
					scores[key] = float32(math.Inf(-1))
					continue
				}
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[row*dim+h*headDim+d] * allK[key*dim+h*headDim+d]
				}
				scores[key] = dot * 0.125
				if scores[key] > maxScore {
					maxScore = scores[key]
				}
			}
			var sum float32
			for key := range scores {
				if !math.IsInf(float64(scores[key]), -1) {
					scores[key] = float32(math.Exp(float64(scores[key] - maxScore)))
					sum += scores[key]
				} else {
					scores[key] = 0
				}
			}
			for key, score := range scores {
				if score == 0 {
					continue
				}
				prob := score / sum
				for d := 0; d < headDim; d++ {
					context[row*dim+h*headDim+d] += prob * allV[key*dim+h*headDim+d]
				}
			}
		}
	}
	gate, err := p.project(name+"gate", x, seq, dim, dim, false)
	if err != nil {
		return nil, err
	}
	for i := range context {
		context[i] *= 1 / (1 + float32(math.Exp(float64(-gate[i]))))
	}
	return p.project(name+"wo", context, seq, dim, dim, false)
}

func (p CheckpointDiT) swiglu(name string, x []float32, rows, dim, hidden int) ([]float32, error) {
	w1, err := p.parameter(name+"w1.weight", hidden, dim)
	if err != nil {
		return nil, err
	}
	w3, err := p.parameter(name+"w3.weight", hidden, dim)
	if err != nil {
		return nil, err
	}
	a, b := linearRows(x, w1, rows, dim, hidden), linearRows(x, w3, rows, dim, hidden)
	for i := range a {
		a[i] = silu(a[i]) * b[i]
	}
	w2, err := p.parameter(name+"w2.weight", dim, hidden)
	if err != nil {
		return nil, err
	}
	return linearRows(a, w2, rows, hidden, dim), nil
}

// Forward returns [latentLen,32] velocity predictions for one sample.
func (p CheckpointDiT) Forward(latent []float32, t float32, text []float32, textMask []bool, speaker []float32, speakerMask []bool, caption []float32, captionMask []bool) ([]float32, error) {
	if p.State == nil || len(latent) == 0 || len(latent)%32 != 0 || len(text) != len(textMask)*512 || len(speaker) != len(speakerMask)*768 || len(caption) != len(captionMask)*512 {
		return nil, fmt.Errorf("irodori: invalid DiT input")
	}
	rows := len(latent) / 32
	time, err := TimestepEmbedding([]float32{t}, 512)
	if err != nil {
		return nil, err
	}
	for i, dim := range []int{1280, 1280, 3840} {
		time, err = p.project(fmt.Sprintf("cond_module.%d", i*2), time, 1, []int{512, 1280, 1280}[i], dim, false)
		if err != nil {
			return nil, err
		}
		if i < 2 {
			for j, v := range time {
				time[j] = silu(v)
			}
		}
	}
	x, err := p.project("in_proj", latent, rows, 32, 1280, true)
	if err != nil {
		return nil, err
	}
	for block := 0; block < 12; block++ {
		base := fmt.Sprintf("blocks.%d.", block)
		h, gate, err := p.adaln(base+"attention_adaln.", x, time)
		if err != nil {
			return nil, err
		}
		attn, err := p.attention(base+"attention.", h, text, speaker, caption, textMask, speakerMask, captionMask)
		if err != nil {
			return nil, err
		}
		for i := range x {
			x[i] += gate[i] * attn[i]
		}
		h, gate, err = p.adaln(base+"mlp_adaln.", x, time)
		if err != nil {
			return nil, err
		}
		mlp, err := p.swiglu(base+"mlp.", h, rows, 1280, 3680)
		if err != nil {
			return nil, err
		}
		for i := range x {
			x[i] += gate[i] * mlp[i]
		}
	}
	w, err := p.parameter("out_norm.weight", 1280)
	if err != nil {
		return nil, err
	}
	x, err = RMSNorm(x, w, 1280, 1e-5)
	if err != nil {
		return nil, err
	}
	return p.project("out_proj", x, rows, 1280, 32, true)
}
