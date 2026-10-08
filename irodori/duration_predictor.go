package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
)

// CheckpointDurationPredictor runs the v4.1-Small token-sum dual AdaRN model.
// The caller supplies encoded text/speaker/caption states; codec and condition
// encoders are separate stages.
type CheckpointDurationPredictor struct{ State *autograd.SafeTensorFile }

func (p CheckpointDurationPredictor) parameter(name string, shape ...int) ([]float32, error) {
	v, s, err := p.State.ReadF32("duration_predictor." + name)
	if err != nil {
		return nil, err
	}
	if len(s) != len(shape) {
		return nil, fmt.Errorf("irodori: duration %s shape %v", name, s)
	}
	for i, d := range shape {
		if s[i] != d {
			return nil, fmt.Errorf("irodori: duration %s shape %v", name, s)
		}
	}
	return v, nil
}

func (p CheckpointDurationPredictor) affine(name string, x []float32, rows, in, out int) ([]float32, error) {
	w, err := p.parameter(name+".weight", out, in)
	if err != nil {
		return nil, err
	}
	b, err := p.parameter(name+".bias", out)
	if err != nil {
		return nil, err
	}
	y := linearRows(x, w, rows, in, out)
	for row := 0; row < rows; row++ {
		for j := 0; j < out; j++ {
			y[row*out+j] += b[j]
		}
	}
	return y, nil
}

func (p CheckpointDurationPredictor) swiglu(name string, x []float32, rows, dim, hidden int) ([]float32, error) {
	w1, err := p.parameter(name+".w1.weight", hidden, dim)
	if err != nil {
		return nil, err
	}
	w3, err := p.parameter(name+".w3.weight", hidden, dim)
	if err != nil {
		return nil, err
	}
	a, b := linearRows(x, w1, rows, dim, hidden), linearRows(x, w3, rows, dim, hidden)
	for i := range a {
		a[i] = silu(a[i]) * b[i]
	}
	w2, err := p.parameter(name+".w2.weight", dim, hidden)
	if err != nil {
		return nil, err
	}
	return linearRows(a, w2, rows, hidden, dim), nil
}

func (p CheckpointDurationPredictor) Forward(text []float32, textMask []bool, speaker []float32, caption []float32, captionMask []bool, hasSpeaker, hasCaption bool) (float32, error) {
	const textDim, hidden = 512, 1024
	if p.State == nil || len(textMask) == 0 || len(text) != len(textMask)*textDim || len(speaker)%768 != 0 || len(caption) != len(captionMask)*512 {
		return 0, fmt.Errorf("irodori: invalid duration input")
	}
	speakerVec := make([]float32, 768)
	if hasSpeaker && len(speaker) >= 768 {
		copy(speakerVec, speaker[:768])
	} else {
		v, err := p.parameter("null_speaker", 768)
		if err != nil {
			return 0, err
		}
		copy(speakerVec, v)
	}
	captionVec := make([]float32, 512)
	if hasCaption {
		count := 0
		for row, valid := range captionMask {
			if valid {
				count++
				for j := 0; j < 512; j++ {
					captionVec[j] += caption[row*512+j]
				}
			}
		}
		if count > 0 {
			for j := range captionVec {
				captionVec[j] /= float32(count)
			}
		} else {
			v, err := p.parameter("null_caption", 512)
			if err != nil {
				return 0, err
			}
			copy(captionVec, v)
		}
	} else {
		v, err := p.parameter("null_caption", 512)
		if err != nil {
			return 0, err
		}
		copy(captionVec, v)
	}
	rows := len(textMask)
	h, err := p.affine("token_input_proj", text, rows, textDim, hidden)
	if err != nil {
		return 0, err
	}
	for block := 0; block < 3; block++ {
		base := fmt.Sprintf("token_blocks.%d.", block)
		w, err := p.parameter(base+"norm.weight", hidden)
		if err != nil {
			return 0, err
		}
		normed, err := RMSNorm(h, w, hidden, 1e-5)
		if err != nil {
			return 0, err
		}
		speakerAct := make([]float32, len(speakerVec))
		captionAct := make([]float32, len(captionVec))
		for i, v := range speakerVec {
			speakerAct[i] = silu(v)
		}
		for i, v := range captionVec {
			captionAct[i] = silu(v)
		}
		sm, err := p.affine(base+"modulation", speakerAct, 1, 768, 3*hidden)
		if err != nil {
			return 0, err
		}
		cm, err := p.affine(base+"caption_modulation", captionAct, 1, 512, 3*hidden)
		if err != nil {
			return 0, err
		}
		for row := 0; row < rows; row++ {
			for j := 0; j < hidden; j++ {
				normed[row*hidden+j] = normed[row*hidden+j]*(1+sm[hidden+j]+cm[hidden+j]) + sm[j] + cm[j]
			}
		}
		out, err := p.swiglu(base+"mlp", normed, rows, hidden, hidden)
		if err != nil {
			return 0, err
		}
		for row := 0; row < rows; row++ {
			for j := 0; j < hidden; j++ {
				h[row*hidden+j] += float32(math.Tanh(float64(sm[2*hidden+j]+cm[2*hidden+j]))) * out[row*hidden+j]
			}
		}
	}
	w, err := p.parameter("token_out_norm.weight", hidden)
	if err != nil {
		return 0, err
	}
	h, err = RMSNorm(h, w, hidden, 1e-5)
	if err != nil {
		return 0, err
	}
	logits, err := p.affine("token_out_proj", h, rows, hidden, 1)
	if err != nil {
		return 0, err
	}
	var frames float32
	for row, v := range logits {
		if textMask[row] {
			frames += float32(math.Log1p(math.Exp(float64(v))))
		}
	}
	return float32(math.Log1p(float64(frames))), nil
}
