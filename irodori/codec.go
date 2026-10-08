package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
)

// DACVAECodec reads a trusted .pth checkpoint after conversion with
// tools/convert_irodori_codec.py. Samples and internal maps are channel-major.
type DACVAECodec struct{ State *autograd.SafeTensorFile }

type audioMap struct {
	data             []float32
	channels, frames int
}

func (c DACVAECodec) parameter(name string, shape ...int) ([]float32, error) {
	v, s, err := c.State.ReadF32(name)
	if err != nil {
		return nil, err
	}
	if len(s) != len(shape) {
		return nil, fmt.Errorf("irodori: codec %s shape %v", name, s)
	}
	for i := range shape {
		if s[i] != shape[i] {
			return nil, fmt.Errorf("irodori: codec %s shape %v", name, s)
		}
	}
	return v, nil
}

func (c DACVAECodec) snake(name string, x audioMap) (audioMap, error) {
	a, err := c.parameter(name+".alpha", 1, x.channels, 1)
	if err != nil {
		return audioMap{}, err
	}
	for ch := 0; ch < x.channels; ch++ {
		alpha := a[ch]
		for i := ch * x.frames; i < (ch+1)*x.frames; i++ {
			v := x.data[i]
			sn := float32(math.Sin(float64(alpha * v)))
			x.data[i] = v + sn*sn/(alpha+1e-9)
		}
	}
	return x, nil
}

func (c DACVAECodec) conv(name string, x audioMap, out, kernel, stride, dilation int) (audioMap, error) {
	w, err := c.parameter(name+".weight", out, x.channels, kernel)
	if err != nil {
		return audioMap{}, err
	}
	b, err := c.parameter(name+".bias", out)
	if err != nil {
		return audioMap{}, err
	}
	pad := (kernel - stride) * dilation / 2
	n := (x.frames+2*pad-dilation*(kernel-1)-1)/stride + 1
	if n <= 0 {
		return audioMap{}, fmt.Errorf("irodori: codec convolution %s short input", name)
	}
	k := x.channels * kernel
	// im2col yields [time, input-channel*kernel], then the shared CPU GEMM
	// handles each channel's dot product with the checkpoint's filter bank.
	cols := make([]float32, n*k)
	for t := 0; t < n; t++ {
		for ch := 0; ch < x.channels; ch++ {
			for tap := 0; tap < kernel; tap++ {
				at := t*stride + tap*dilation - pad
				if at >= 0 && at < x.frames {
					cols[t*k+ch*kernel+tap] = x.data[ch*x.frames+at]
				}
			}
		}
	}
	y := make([]float32, n*out)
	tensor.SGEMMOp(y, cols, w, n, out, k, false, true)
	result := audioMap{data: make([]float32, n*out), channels: out, frames: n}
	for ch := 0; ch < out; ch++ {
		for t := 0; t < n; t++ {
			result.data[ch*n+t] = y[t*out+ch] + b[ch]
		}
	}
	return result, nil
}

func (c DACVAECodec) residual(name string, x audioMap, dilation int) (audioMap, error) {
	h, err := c.snake(name+".block.0", audioMap{data: append([]float32(nil), x.data...), channels: x.channels, frames: x.frames})
	if err != nil {
		return audioMap{}, err
	}
	h, err = c.conv(name+".block.1", h, x.channels, 7, 1, dilation)
	if err != nil {
		return audioMap{}, err
	}
	h, err = c.snake(name+".block.2", h)
	if err != nil {
		return audioMap{}, err
	}
	h, err = c.conv(name+".block.3", h, x.channels, 1, 1, 1)
	if err != nil {
		return audioMap{}, err
	}
	if h.frames != x.frames {
		return audioMap{}, fmt.Errorf("irodori: residual length %d != %d", h.frames, x.frames)
	}
	for i := range h.data {
		h.data[i] += x.data[i]
	}
	return h, nil
}

// Encode48k normalizes a mono waveform and returns time-major [frames,32]
// deterministic DACVAE posterior means, as used for Irodori references.
func (c DACVAECodec) Encode48k(wav []float32) ([]float32, error) {
	if c.State == nil {
		return nil, fmt.Errorf("irodori: missing codec state")
	}
	norm, err := NormalizeReference48k(wav)
	if err != nil {
		return nil, err
	}
	pad := (1920 - len(norm)%1920) % 1920
	if pad > len(norm) {
		return nil, fmt.Errorf("irodori: reference too short for reflection padding")
	}
	for i := 0; i < pad; i++ {
		norm = append(norm, norm[len(wav)-2-i])
	}
	x := audioMap{data: norm, channels: 1, frames: len(norm)}
	x, err = c.conv("encoder.block.0", x, 64, 7, 1, 1)
	if err != nil {
		return nil, err
	}
	channels := 64
	for block, stride := range []int{2, 8, 10, 12} {
		base := fmt.Sprintf("encoder.block.%d.block.", block+1)
		for unit, dilation := range []int{1, 3, 9} {
			x, err = c.residual(fmt.Sprintf("%s%d", base, unit), x, dilation)
			if err != nil {
				return nil, err
			}
		}
		x, err = c.snake(base+"3", x)
		if err != nil {
			return nil, err
		}
		x, err = c.conv(base+"4", x, channels*2, 2*stride, stride, 1)
		if err != nil {
			return nil, err
		}
		channels *= 2
	}
	x, err = c.snake("encoder.block.5", x)
	if err != nil {
		return nil, err
	}
	x, err = c.conv("encoder.block.6", x, 1024, 3, 1, 1)
	if err != nil {
		return nil, err
	}
	x, err = c.conv("quantizer.in_proj", x, 64, 1, 1, 1)
	if err != nil {
		return nil, err
	}
	out := make([]float32, x.frames*32)
	for t := 0; t < x.frames; t++ {
		for ch := 0; ch < 32; ch++ {
			out[t*32+ch] = x.data[ch*x.frames+t]
		}
	}
	return out, nil
}

func (c DACVAECodec) convTranspose(name string, x audioMap, out, kernel, stride int) (audioMap, error) {
	w, err := c.parameter(name+".weight", x.channels, out, kernel)
	if err != nil {
		return audioMap{}, err
	}
	b, err := c.parameter(name+".bias", out)
	if err != nil {
		return audioMap{}, err
	}
	pad := (stride + 1) / 2
	outputPad := stride % 2
	frames := (x.frames-1)*stride - 2*pad + kernel + outputPad
	if frames <= 0 {
		return audioMap{}, fmt.Errorf("irodori: transposed convolution short input")
	}
	rows := make([]float32, x.frames*x.channels)
	for t := 0; t < x.frames; t++ {
		for ch := 0; ch < x.channels; ch++ {
			rows[t*x.channels+ch] = x.data[ch*x.frames+t]
		}
	}
	cols := make([]float32, x.frames*out*kernel)
	// Weight layout is [in,out,kernel], so no transpose is needed.
	tensor.SGEMM(cols, rows, w, x.frames, out*kernel, x.channels)
	y := audioMap{data: make([]float32, out*frames), channels: out, frames: frames}
	for ch := 0; ch < out; ch++ {
		for t := 0; t < frames; t++ {
			y.data[ch*frames+t] = b[ch]
		}
	}
	for t := 0; t < x.frames; t++ {
		for ch := 0; ch < out; ch++ {
			for tap := 0; tap < kernel; tap++ {
				at := t*stride + tap - pad
				if at >= 0 && at < frames {
					y.data[ch*frames+at] += cols[t*out*kernel+ch*kernel+tap]
				}
			}
		}
	}
	return y, nil
}

// Decode converts time-major [frames,32] codec latents into raw 48 kHz audio.
// It follows the reference's deterministic alpha=0 decoder path; a separate
// SilentCipher pass remains required before any WAV output.
func (c DACVAECodec) Decode(latent []float32) ([]float32, error) {
	if c.State == nil || len(latent) == 0 || len(latent)%32 != 0 {
		return nil, fmt.Errorf("irodori: invalid decoder latent")
	}
	frames := len(latent) / 32
	x := audioMap{data: make([]float32, 32*frames), channels: 32, frames: frames}
	for t := 0; t < frames; t++ {
		for ch := 0; ch < 32; ch++ {
			x.data[ch*frames+t] = latent[t*32+ch]
		}
	}
	var err error
	x, err = c.conv("quantizer.out_proj", x, 1024, 1, 1, 1)
	if err != nil {
		return nil, err
	}
	x, err = c.conv("decoder.model.0", x, 1536, 7, 1, 1)
	if err != nil {
		return nil, err
	}
	for block, stride := range []int{12, 10, 8, 2} {
		base := fmt.Sprintf("decoder.model.%d.block.", block+1)
		x, err = c.snake(base+"0", x)
		if err != nil {
			return nil, err
		}
		x, err = c.convTranspose(base+"1", x, x.channels/2, 2*stride, stride)
		if err != nil {
			return nil, err
		}
		for _, unit := range []struct{ index, dilation int }{{4, 1}, {5, 3}, {8, 9}} {
			x, err = c.residual(fmt.Sprintf("%s%d", base, unit.index), x, unit.dilation)
			if err != nil {
				return nil, err
			}
		}
	}
	x, err = c.snake("decoder.wm_model.encoder_block.pre.0", x)
	if err != nil {
		return nil, err
	}
	x, err = c.conv("decoder.wm_model.encoder_block.pre.1", x, 1, 7, 1, 1)
	if err != nil {
		return nil, err
	}
	// forward_no_conv then applies Tanh and skips the last watermark conv.
	for i := range x.data {
		x.data[i] = float32(math.Tanh(float64(x.data[i])))
	}
	return x.data, nil
}
