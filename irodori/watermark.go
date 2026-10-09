package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/dsp"
	"github.com/yh2237/gograd/tensor"
)

const (
	// watermarkPayload is the reference's message, ASCII for "IRDTS".
	watermarkPayload = "IRDTS"
	// watermarkAverageEnergy is the VCTK reference energy the model normalizes
	// to before encoding, and away from afterwards.
	watermarkAverageEnergy = 0.002837200844477648
	// watermarkMessageSDR is the model's configured message-to-carrier ratio.
	watermarkMessageSDR = 47

	// The rest come from the model's hparams.yaml and the dimensions server.py
	// hardcodes.
	watermarkFilterLength = 4096
	watermarkHopLength    = 2048
	watermarkSampleRate   = 44100
	watermarkMessageDim   = 5
	watermarkMessageLen   = 21
	watermarkMessageBand  = 1024
	watermarkEncoderOut   = 32
	watermarkDecoderIn    = watermarkEncoderOut * 3
	watermarkEpsilon      = 1e-5
)

// grid is one sample's channel-major (channels, freq, time) block, which is the
// reference's tensor layout once the batch dimension is dropped. Indexing it
// this way lets the convolution weights keep the checkpoint's layout.
type grid struct {
	data                 []float32
	channels, bins, rows int
}

func (g grid) at(channel, bin, frame int) int { return (channel*g.bins+bin)*g.rows + frame }

// workspace holds the scratch buffers a forward pass reuses. A production grid
// is tens of megabytes per intermediate, so one growable set avoids churning the
// heap between layers and between taps.
type workspace struct {
	padded   []float32
	tapPatch []float32
	tapWt    []float32
	partial  []float32
}

func (w *workspace) buffer(slot *[]float32, n int) []float32 {
	if cap(*slot) < n {
		*slot = make([]float32, n)
	}
	return (*slot)[:n]
}

// conv2d convolves a grid with a zero-padded window. The kernel size is odd and
// the padding keeps the frequency and time axes unchanged, which is all the
// watermark's layers use.
type conv2d struct {
	weight                       []float32
	bias                         []float32
	inChannels, outChans, kernel int
}

// forward accumulates one shifted matrix product per kernel tap. Materializing
// the im2col matrix of a padded 96-channel grid would need hundreds of
// megabytes, and a tap only needs one neighbour offset of the input, so the
// offsets are folded into small products over a channel-major patch instead.
func (c conv2d) forward(in grid, ws *workspace) (grid, error) {
	if in.channels != c.inChannels {
		return grid{}, fmt.Errorf("irodori: watermark convolution expects %d channels, got %d", c.inChannels, in.channels)
	}
	pad := c.kernel / 2
	// A padded channel holds the input rows plus pad zero rows on each side.
	paddedRows := in.rows + 2*pad
	stride := (in.bins + 2*pad) * paddedRows
	padded := ws.buffer(&ws.padded, in.channels*stride)
	for i := range padded {
		padded[i] = 0
	}
	for channel := 0; channel < in.channels; channel++ {
		src, dst := channel*in.bins*in.rows, channel*stride
		for bin := 0; bin < in.bins; bin++ {
			copy(padded[dst+(bin+pad)*paddedRows+pad:], in.data[src+bin*in.rows:src+bin*in.rows+in.rows])
		}
	}
	columns := in.bins * in.rows
	tapPatch := ws.buffer(&ws.tapPatch, in.channels*columns)
	tapWt := ws.buffer(&ws.tapWt, c.outChans*c.inChannels)
	partial := ws.buffer(&ws.partial, c.outChans*columns)
	out := grid{data: make([]float32, c.outChans*columns), channels: c.outChans, bins: in.bins, rows: in.rows}
	for i := range out.data {
		out.data[i] = 0
	}
	for ky := 0; ky < c.kernel; ky++ {
		for kx := 0; kx < c.kernel; kx++ {
			// Gather this offset. A patch row is one contiguous frequency bin, so
			// each row is a copy rather than a per-element gather.
			for channel := 0; channel < in.channels; channel++ {
				src, dst := channel*stride, channel*columns
				for bin := 0; bin < in.bins; bin++ {
					from := src + (bin+ky)*paddedRows + kx
					copy(tapPatch[dst+bin*in.rows:], padded[from:from+in.rows])
				}
			}
			// The weight is (out, in, kernel, kernel), so a tap slices the last
			// two axes into an (out, in) matrix.
			for outChan := 0; outChan < c.outChans; outChan++ {
				for inChan := 0; inChan < c.inChannels; inChan++ {
					tapWt[outChan*c.inChannels+inChan] = c.weight[((outChan*c.inChannels+inChan)*c.kernel+ky)*c.kernel+kx]
				}
			}
			tensor.SGEMMOp(partial, tapWt, tapPatch, c.outChans, columns, c.inChannels, false, false)
			for i := range out.data {
				out.data[i] += partial[i]
			}
		}
	}
	for outChan := 0; outChan < c.outChans; outChan++ {
		bias := c.bias[outChan]
		for i := outChan * columns; i < (outChan+1)*columns; i++ {
			out.data[i] += bias
		}
	}
	return out, nil
}

// layer is one of the reference's gated convolutions: two convolutions with the
// same shape, a sigmoid on the second, and a batch normalization applied after
// their product.
//
// The reference never calls eval() on the watermark model, so the normalization
// uses the batch statistics of its own input rather than the checkpoint's
// running statistics. That also makes the watermark's output independent of any
// earlier utterance, because a running-statistic update never feeds back into
// the forward pass.
type layer struct {
	conv  conv2d
	gate  conv2d
	gamma []float32
	beta  []float32
}

func (l layer) forward(in grid, ws *workspace) (grid, error) {
	main, err := l.conv.forward(in, ws)
	if err != nil {
		return grid{}, err
	}
	gated, err := l.gate.forward(in, ws)
	if err != nil {
		return grid{}, err
	}
	for i := range main.data {
		main.data[i] *= sigmoid(gated.data[i])
	}
	l.normalize(main)
	return main, nil
}

// normalize replaces each channel with its standardized form, scaling by the
// learned gamma and shifting by beta. The mean over the frequency and time axes
// is subtracted first and the biased variance is then taken around it, which is
// the two-pass form the reference's Welford accumulation matches.
func (l layer) normalize(g grid) {
	block := g.bins * g.rows
	centered := make([]float32, block)
	for channel := 0; channel < g.channels; channel++ {
		from := channel * block
		var sum float64
		for i, v := range g.data[from : from+block] {
			centered[i] = v
			sum += float64(v)
		}
		mean := float32(sum / float64(block))
		var sumSq float64
		for _, v := range centered {
			sumSq += float64(v-mean) * float64(v-mean)
		}
		variance := float32(sumSq / float64(block))
		scale := l.gamma[channel] / float32(math.Sqrt(float64(variance+watermarkEpsilon)))
		shift := l.beta[channel] - mean*scale
		for i := range centered {
			g.data[from+i] = centered[i]*scale + shift
		}
	}
}

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// layerSpec is one gated convolution's channel counts and kernel size.
type layerSpec struct {
	in, out, kernel int
}

// watermarkEncoderLayers is the encoder's stack: a 1-to-32 layer followed by two
// at the encoder's output width.
var watermarkEncoderLayers = []layerSpec{
	{1, watermarkEncoderOut, 3},
	{watermarkEncoderOut, watermarkEncoderOut, 3},
	{watermarkEncoderOut, watermarkEncoderOut, 3},
}

// watermarkDecoderLayers is the carrier decoder's stack: three layers at the
// concatenated width, then a one-by-one projection to a single channel.
var watermarkDecoderLayers = []layerSpec{
	{watermarkDecoderIn, watermarkDecoderIn, 3},
	{watermarkDecoderIn, watermarkDecoderIn, 3},
	{watermarkDecoderIn, watermarkDecoderIn, 3},
	{watermarkDecoderIn, 1, 1},
}

// SilentCipher is the reference's audio watermark. It works in the magnitude
// STFT domain rather than on samples: the carrier spectrum is encoded, the
// message is expanded into a per-bin bias, and a carrier decoder predicts the
// magnitude perturbation that carries the message.
//
// It needs the converted checkpoint written by tools/convert_silentcipher.py.
type SilentCipher struct {
	State     *autograd.SafeTensorFile
	spectral  *dsp.STFT
	encLayers []layer
	decLayers []layer
	msgWeight []float32
	msgBias   []float32
	bins      int
}

// LoadSilentCipher reads a converted watermark checkpoint and validates every
// tensor shape against the architecture the reference builds.
func LoadSilentCipher(state *autograd.SafeTensorFile) (*SilentCipher, error) {
	if state == nil {
		return nil, fmt.Errorf("irodori: missing watermark checkpoint")
	}
	spectral, err := dsp.NewSTFT(watermarkFilterLength, watermarkHopLength)
	if err != nil {
		return nil, err
	}
	c := &SilentCipher{State: state, spectral: spectral, bins: spectral.Bins()}
	for index, spec := range watermarkEncoderLayers {
		l, err := c.layer(fmt.Sprintf("enc_c.main.%d", index), spec)
		if err != nil {
			return nil, err
		}
		c.encLayers = append(c.encLayers, l)
	}
	for index, spec := range watermarkDecoderLayers {
		l, err := c.layer(fmt.Sprintf("dec_c.main.%d", index), spec)
		if err != nil {
			return nil, err
		}
		c.decLayers = append(c.decLayers, l)
	}
	if c.msgWeight, err = c.parameter("enc_c.linear.weight", watermarkMessageBand, watermarkMessageDim); err != nil {
		return nil, err
	}
	if c.msgBias, err = c.parameter("enc_c.linear.bias", watermarkMessageBand); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *SilentCipher) parameter(name string, shape ...int) ([]float32, error) {
	v, s, err := c.State.ReadF32(name)
	if err != nil {
		return nil, err
	}
	if len(s) != len(shape) {
		return nil, fmt.Errorf("irodori: watermark %s shape %v", name, s)
	}
	for i := range shape {
		if s[i] != shape[i] {
			return nil, fmt.Errorf("irodori: watermark %s shape %v", name, s)
		}
	}
	return v, nil
}

func (c *SilentCipher) layer(name string, spec layerSpec) (layer, error) {
	conv, err := c.convolution(name+".conv", spec)
	if err != nil {
		return layer{}, err
	}
	gate, err := c.convolution(name+".gate", spec)
	if err != nil {
		return layer{}, err
	}
	// The running statistics are still loaded, because the reference updates
	// them with the batch statistics of every call, but the forward pass in
	// training mode never reads them.
	gamma, err := c.parameter(name+".bn.weight", spec.out)
	if err != nil {
		return layer{}, err
	}
	beta, err := c.parameter(name+".bn.bias", spec.out)
	if err != nil {
		return layer{}, err
	}
	if _, err = c.parameter(name+".bn.running_mean", spec.out); err != nil {
		return layer{}, err
	}
	if _, err = c.parameter(name+".bn.running_var", spec.out); err != nil {
		return layer{}, err
	}
	return layer{conv: conv, gate: gate, gamma: gamma, beta: beta}, nil
}

func (c *SilentCipher) convolution(name string, spec layerSpec) (conv2d, error) {
	weight, err := c.parameter(name+".weight", spec.out, spec.in, spec.kernel, spec.kernel)
	if err != nil {
		return conv2d{}, err
	}
	bias, err := c.parameter(name+".bias", spec.out)
	if err != nil {
		return conv2d{}, err
	}
	return conv2d{weight: weight, bias: bias, inChannels: spec.in, outChans: spec.out, kernel: spec.kernel}, nil
}

// ExpandMessage returns the (message_dim, frames) one-hot grid the reference
// tiles across the spectrum's time axis. The payload's bytes become bits, taken
// two at a time, each value incremented by one, and a zero terminator appended;
// the resulting 21-symbol pattern repeats once per frame.
func ExpandMessage(payload []byte, messageDim, messageLen, frames int) ([]float32, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf("irodori: empty watermark payload")
	}
	var bits string
	for _, b := range payload {
		bits += fmt.Sprintf("%08b", b)
	}
	const bitsPerSymbol = 2
	groups := len(bits) / bitsPerSymbol
	if groups+1 != messageLen {
		return nil, fmt.Errorf("irodori: payload of %d bytes gives %d symbols, want %d", len(payload), groups+1, messageLen)
	}
	pattern := make([]int, messageLen)
	for i := 0; i < groups; i++ {
		value := int(bits[i*bitsPerSymbol]-'0')*2 + int(bits[i*bitsPerSymbol+1]-'0')
		pattern[i] = value + 1
	}
	pattern[groups] = 0
	out := make([]float32, messageDim*frames)
	for frame := 0; frame < frames; frame++ {
		out[pattern[frame%messageLen]*frames+frame] = 1
	}
	return out, nil
}

// transformMessage applies the checkpoint's message projection along the symbol
// axis, then pads the frequency axis out to the spectrum's bin count. The
// reference keeps the message in the lowest message_band_size bins.
func (c *SilentCipher) transformMessage(message []float32, frames int) []float32 {
	expanded := make([]float32, watermarkMessageBand*frames)
	tensor.SGEMMOp(expanded, c.msgWeight, message, watermarkMessageBand, frames, watermarkMessageDim, false, false)
	for k := 0; k < watermarkMessageBand; k++ {
		bias := c.msgBias[k]
		for frame := 0; frame < frames; frame++ {
			expanded[k*frames+frame] += bias
		}
	}
	out := make([]float32, c.bins*frames)
	copy(out, expanded)
	return out
}

// Encode returns audio with the watermark applied. The model runs at its own
// sample rate, so other rates are resampled down and back and the result is
// trimmed to the caller's length, exactly as the reference does.
func (c *SilentCipher) Encode(audio []float32, sampleRate int) ([]float32, error) {
	if len(audio) == 0 {
		return nil, fmt.Errorf("irodori: empty watermark input")
	}
	signal := audio
	if sampleRate != watermarkSampleRate {
		signal = dsp.Resample(audio, sampleRate, watermarkSampleRate)
	}
	out := c.encode(signal)
	if sampleRate != watermarkSampleRate {
		out = dsp.Resample(out, watermarkSampleRate, sampleRate)
		if len(out) > len(audio) {
			out = out[:len(audio)]
		}
	}
	return out, nil
}

// watermarkStages holds the spectra a parity test compares one stage at a time,
// so a divergence localizes to the transform, the carrier encoder, the carrier
// decoder or the perturbation arithmetic.
type watermarkStages struct {
	carrier, encoded, merged, perturbation, rebuilt []float32
	frames                                          int
}

// encodeParts runs the model's spectral stages on a signal that is already at
// the model's sample rate, returning every intermediate grid.
func (c *SilentCipher) encodeParts(signal []float32) watermarkStages {
	var power float64
	for _, v := range signal {
		power += float64(v) * float64(v)
	}
	power /= float64(len(signal))
	if power == 0 {
		return watermarkStages{}
	}
	gain := float32(math.Sqrt(watermarkAverageEnergy / power))
	scaled := make([]float32, len(signal))
	for i, v := range signal {
		scaled[i] = v * gain
	}
	magnitude, _, frames := c.spectral.Transform(scaled)
	if frames == 0 {
		return watermarkStages{}
	}
	carrierGrid := grid{data: magnitude, channels: 1, bins: c.bins, rows: frames}
	message, err := ExpandMessage([]byte(watermarkPayload), watermarkMessageDim, watermarkMessageLen, frames)
	if err != nil {
		return watermarkStages{}
	}
	ws := &workspace{}
	encoded := carrierGrid
	for _, l := range c.encLayers {
		encoded, err = l.forward(encoded, ws)
		if err != nil {
			return watermarkStages{}
		}
	}
	messageGrid := c.transformMessage(message, frames)
	block := c.bins * frames
	merged := grid{data: make([]float32, watermarkDecoderIn*block), channels: watermarkDecoderIn, bins: c.bins, rows: frames}
	copy(merged.data, encoded.data)
	for channel := 0; channel < watermarkEncoderOut; channel++ {
		copy(merged.data[watermarkEncoderOut*block+channel*block:], carrierGrid.data)
	}
	for channel := 0; channel < watermarkEncoderOut; channel++ {
		copy(merged.data[2*watermarkEncoderOut*block+channel*block:], messageGrid)
	}
	perturbation := merged
	for _, l := range c.decLayers {
		perturbation, err = l.forward(perturbation, ws)
		if err != nil {
			return watermarkStages{}
		}
	}
	perturbation = grid{data: c.decoderPostProcess(perturbation), channels: 1, bins: c.bins, rows: frames}
	// Utterance-level normalization scales the perturbation to the carrier's own
	// energy before it is negated and clamped into the carrier.
	var carrierSum float64
	for _, v := range carrierGrid.data {
		carrierSum += float64(v) * float64(v)
	}
	scale := float32(math.Sqrt(carrierSum / float64(len(carrierGrid.data))))
	rebuilt := make([]float32, len(carrierGrid.data))
	for i := range rebuilt {
		rebuilt[i] = max(0, carrierGrid.data[i]-scale*perturbation.data[i])
	}
	return watermarkStages{carrier: carrierGrid.data, encoded: encoded.data,
		merged: merged.data, perturbation: perturbation.data, rebuilt: rebuilt, frames: frames}
}

// encode is the model's own pass over a signal already at its sample rate. It
// repeats the carrier's normalization so the phase used by the inverse always
// matches the spectrum that was perturbed.
func (c *SilentCipher) encode(signal []float32) []float32 {
	stages := c.encodeParts(signal)
	if stages.frames == 0 {
		return append([]float32(nil), signal...)
	}
	var power float64
	for _, v := range signal {
		power += float64(v) * float64(v)
	}
	power /= float64(len(signal))
	gain := float32(math.Sqrt(watermarkAverageEnergy / power))
	scaled := make([]float32, len(signal))
	for i, v := range signal {
		scaled[i] = v * gain
	}
	_, phase, _ := c.spectral.Transform(scaled)
	out := c.spectral.Inverse(stages.rebuilt, phase, len(signal))
	restore := float32(math.Sqrt(power / watermarkAverageEnergy))
	for i := range out {
		out[i] *= restore
	}
	return out
}

func (c *SilentCipher) decoderPostProcess(g grid) []float32 {
	out := append([]float32(nil), g.data...)
	for i := range out {
		out[i] = float32(math.Abs(float64(out[i])))
	}
	for bin := watermarkMessageBand; bin < c.bins; bin++ {
		for frame := 0; frame < g.rows; frame++ {
			out[g.at(0, bin, frame)] = 0
		}
	}
	// Each frame is normalized by the root mean square of its frequency axis,
	// scaled to the configured message-to-carrier ratio. The reference writes
	// 10**(message_sdr/20) in true division, so the ratio is in decibels.
	divisor := float32(math.Pow(10, float64(watermarkMessageSDR)/20))
	for frame := 0; frame < g.rows; frame++ {
		var sum float64
		for bin := 0; bin < c.bins; bin++ {
			v := float64(out[g.at(0, bin, frame)])
			sum += v * v
		}
		rms := float32(math.Sqrt(sum/float64(c.bins))) * divisor
		if rms == 0 {
			continue
		}
		for bin := 0; bin < c.bins; bin++ {
			out[g.at(0, bin, frame)] /= rms
		}
	}
	return out
}
