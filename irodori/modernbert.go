package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
)

// ModernBERT reads the shared 25-layer Japanese text backbone from the real
// Irodori safetensors checkpoint, one layer at a time. It is inference-only.
type ModernBERT struct{ State *autograd.SafeTensorFile }

const modernPrefix = "pretrained_text_backbone.backbone."

func (m ModernBERT) param(name string, shape ...int) ([]float32, error) {
	v, s, err := m.State.ReadF32(modernPrefix + name)
	if err != nil {
		return nil, err
	}
	if len(s) != len(shape) {
		return nil, fmt.Errorf("irodori: %s shape %v", name, s)
	}
	for i, d := range shape {
		if s[i] != d {
			return nil, fmt.Errorf("irodori: %s shape %v", name, s)
		}
	}
	return v, nil
}

func linearRows(x, w []float32, rows, in, out int) []float32 {
	y := make([]float32, rows*out)
	tensor.SGEMMOp(y, x, w, rows, out, in, false, true)
	return y
}

func layerNormRows(x, w []float32, rows, width int) []float32 {
	y := make([]float32, len(x))
	for row := 0; row < rows; row++ {
		start := row * width
		var mean, variance float32
		for _, v := range x[start : start+width] {
			mean += v
		}
		mean /= float32(width)
		for _, v := range x[start : start+width] {
			d := v - mean
			variance += d * d
		}
		variance /= float32(width)
		inv := float32(1 / math.Sqrt(float64(variance+1e-5)))
		for j := 0; j < width; j++ {
			y[start+j] = (x[start+j] - mean) * inv * w[j]
		}
	}
	return y
}

func modernAttention(x, qkv, wo []float32, mask []bool, sliding bool, theta float64) []float32 {
	const width, heads, headDim = 768, 12, 64
	seq := len(mask)
	all := linearRows(x, qkv, seq, width, 3*width)
	q, k, v := make([]float32, seq*width), make([]float32, seq*width), make([]float32, seq*width)
	for pos := 0; pos < seq; pos++ {
		copy(q[pos*width:(pos+1)*width], all[pos*3*width:pos*3*width+width])
		copy(k[pos*width:(pos+1)*width], all[pos*3*width+width:pos*3*width+2*width])
		copy(v[pos*width:(pos+1)*width], all[pos*3*width+2*width:(pos+1)*3*width])
	}
	for pos := 0; pos < seq; pos++ {
		for h := 0; h < heads; h++ {
			for d := 0; d < headDim/2; d++ {
				freq := float32(1 / math.Pow(theta, float64(float32(2*d)/headDim)))
				angle := float32(pos) * freq
				co, si := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
				i := pos*width + h*headDim + d
				j := i + headDim/2
				qa, qb, ka, kb := q[i], q[j], k[i], k[j]
				q[i], q[j] = qa*co-qb*si, qb*co+qa*si
				k[i], k[j] = ka*co-kb*si, kb*co+ka*si
			}
		}
	}
	context := make([]float32, seq*width)
	for pos := 0; pos < seq; pos++ {
		for h := 0; h < heads; h++ {
			scores := make([]float32, seq)
			maxScore := float32(math.Inf(-1))
			for key := 0; key < seq; key++ {
				if !mask[key] || (sliding && (key < pos-64 || key > pos+64)) {
					scores[key] = float32(math.Inf(-1))
					continue
				}
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[pos*width+h*headDim+d] * k[key*width+h*headDim+d]
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
			if sum == 0 {
				continue
			}
			for key := range scores {
				p := scores[key] / sum
				if p == 0 {
					continue
				}
				for d := 0; d < headDim; d++ {
					context[pos*width+h*headDim+d] += p * v[key*width+h*headDim+d]
				}
			}
		}
	}
	return linearRows(context, wo, seq, width, width)
}

// Forward returns [sequence,768] masked hidden states. IDs and masks must
// already follow the tokenizer wrapper's BOS and right-padding rules.
func (m ModernBERT) Forward(ids []int, mask []bool) ([]float32, error) {
	const width = 768
	if m.State == nil || len(ids) == 0 || len(ids) != len(mask) {
		return nil, fmt.Errorf("irodori: invalid ModernBERT input")
	}
	emb, err := m.param("embeddings.tok_embeddings.weight", 102400, width)
	if err != nil {
		return nil, err
	}
	x := make([]float32, len(ids)*width)
	for i, id := range ids {
		if id < 0 || id >= 102400 {
			return nil, fmt.Errorf("irodori: token id %d", id)
		}
		copy(x[i*width:(i+1)*width], emb[id*width:(id+1)*width])
	}
	emb = nil
	norm, err := m.param("embeddings.norm.weight", width)
	if err != nil {
		return nil, err
	}
	x = layerNormRows(x, norm, len(ids), width)
	for layer := 0; layer < 25; layer++ {
		base := fmt.Sprintf("layers.%d.", layer)
		input := x
		if layer > 0 {
			w, e := m.param(base+"attn_norm.weight", width)
			if e != nil {
				return nil, e
			}
			input = layerNormRows(x, w, len(ids), width)
		}
		qkv, e := m.param(base+"attn.Wqkv.weight", 3*width, width)
		if e != nil {
			return nil, e
		}
		wo, e := m.param(base+"attn.Wo.weight", width, width)
		if e != nil {
			return nil, e
		}
		sliding := layer%3 != 0
		theta := 160000.0
		if sliding {
			theta = 10000
		}
		attn := modernAttention(input, qkv, wo, mask, sliding, theta)
		for i := range x {
			x[i] += attn[i]
		}
		nw, e := m.param(base+"mlp_norm.weight", width)
		if e != nil {
			return nil, e
		}
		input = layerNormRows(x, nw, len(ids), width)
		wi, e := m.param(base+"mlp.Wi.weight", 6144, width)
		if e != nil {
			return nil, e
		}
		up := linearRows(input, wi, len(ids), width, 6144)
		for row := range ids {
			for j := 0; j < 3072; j++ {
				v := up[row*6144+j]
				gelu := float32(0.5) * v * (1 + float32(math.Erf(float64(v)/math.Sqrt2)))
				up[row*6144+j] = gelu * up[row*6144+3072+j]
			}
		}
		compact := make([]float32, len(ids)*3072)
		for row := range ids {
			copy(compact[row*3072:(row+1)*3072], up[row*6144:row*6144+3072])
		}
		mw, e := m.param(base+"mlp.Wo.weight", width, 3072)
		if e != nil {
			return nil, e
		}
		mlp := linearRows(compact, mw, len(ids), 3072, width)
		for i := range x {
			x[i] += mlp[i]
		}
	}
	fw, err := m.param("final_norm.weight", width)
	if err != nil {
		return nil, err
	}
	x = layerNormRows(x, fw, len(ids), width)
	for i, valid := range mask {
		if !valid {
			clear(x[i*width : (i+1)*width])
		}
	}
	return x, nil
}

// Project maps shared ModernBERT states to the checkpoint's text or caption
// context width, including its residual MLP and final RMSNorm.
func (m ModernBERT) Project(hidden []float32, mask []bool, kind string) ([]float32, error) {
	if kind != "text" && kind != "caption" {
		return nil, fmt.Errorf("irodori: invalid condition kind %q", kind)
	}
	if len(hidden) != len(mask)*768 {
		return nil, fmt.Errorf("irodori: invalid condition shape")
	}
	read := func(name string, shape ...int) ([]float32, error) {
		v, s, err := m.State.ReadF32(name)
		if err != nil {
			return nil, err
		}
		if len(s) != len(shape) {
			return nil, fmt.Errorf("irodori: %s shape %v", name, s)
		}
		for i, d := range shape {
			if s[i] != d {
				return nil, fmt.Errorf("irodori: %s shape %v", name, s)
			}
		}
		return v, nil
	}
	base := kind + "_encoder."
	w, err := read(base+"projector.weight", 512, 768)
	if err != nil {
		return nil, err
	}
	b, err := read(base+"projector.bias", 512)
	if err != nil {
		return nil, err
	}
	projected := linearRows(hidden, w, len(mask), 768, 512)
	for row := range mask {
		for j := 0; j < 512; j++ {
			projected[row*512+j] += b[j]
		}
	}
	nw, err := read(base+"residual_norm.weight", 768)
	if err != nil {
		return nil, err
	}
	normed, err := RMSNorm(hidden, nw, 768, 1e-5)
	if err != nil {
		return nil, err
	}
	upW, err := read(base+"residual_up.weight", 1024, 768)
	if err != nil {
		return nil, err
	}
	upB, err := read(base+"residual_up.bias", 1024)
	if err != nil {
		return nil, err
	}
	up := linearRows(normed, upW, len(mask), 768, 1024)
	for row := range mask {
		for j := 0; j < 1024; j++ {
			up[row*1024+j] = silu(up[row*1024+j] + upB[j])
		}
	}
	downW, err := read(base+"residual_down.weight", 512, 1024)
	if err != nil {
		return nil, err
	}
	downB, err := read(base+"residual_down.bias", 512)
	if err != nil {
		return nil, err
	}
	residual := linearRows(up, downW, len(mask), 1024, 512)
	for row, valid := range mask {
		for j := 0; j < 512; j++ {
			if valid {
				projected[row*512+j] += residual[row*512+j] + downB[j]
			} else {
				projected[row*512+j] = 0
			}
		}
	}
	finalW, err := read(kind+"_norm.weight", 512)
	if err != nil {
		return nil, err
	}
	return RMSNorm(projected, finalW, 512, 1e-5)
}
