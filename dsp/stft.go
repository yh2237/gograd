package dsp

import "math"

// HannPeriodic returns the periodic Hann window torch.hann_window(n) uses. The
// reference builds it once and shares it between the transform and its inverse,
// so both directions must agree on exactly these values.
func HannPeriodic(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// STFT mirrors the reference's magnitude/phase spectral transform. It is not a
// general short-time transform: the input is padded up to a multiple of the
// window, centred with reflect padding exactly as torch.stft does, and the
// inverse trims the same multiple away again.
//
// Spectra are stored frequency major, [bins][frames], matching torch.stft's
// (freq, time) layout so they can feed a two-dimensional convolution without a
// transpose.
type STFT struct {
	fft      *FFT
	window   []float32
	windowSq []float32
	hop      int
	center   int
}

// NewSTFT returns a transform with the reference's filter length and hop.
func NewSTFT(filterLength, hopLength int) (*STFT, error) {
	f, err := NewFFT(filterLength)
	if err != nil {
		return nil, err
	}
	window := HannPeriodic(filterLength)
	windowSq := make([]float32, len(window))
	for i, v := range window {
		windowSq[i] = v * v
	}
	return &STFT{fft: f, window: window, windowSq: windowSq, hop: hopLength, center: filterLength / 2}, nil
}

// Bins reports the number of frequency bins in one frame.
func (s *STFT) Bins() int { return s.fft.Bins() }

// FramesFor reports how many frames Transform produces for a block of length
// samples, including the reflect centring torch.stft adds.
func (s *STFT) FramesFor(length int) int {
	return 1 + padLength(length, len(s.window))/s.hop
}

// Transform returns the frequency-major magnitude and phase of x and the frame
// count. Magnitude uses the reference's epsilon guard, so an exactly-zero bin
// reports a zero magnitude rather than sqrt(epsilon).
func (s *STFT) Transform(x []float32) (magnitude, phase []float32, frames int) {
	padded := s.pad(x)
	frames = 1 + (len(padded)-len(s.window))/s.hop
	bins := s.fft.Bins()
	magnitude = make([]float32, bins*frames)
	phase = make([]float32, bins*frames)
	frame := make([]float32, len(s.window))
	for f := 0; f < frames; f++ {
		copy(frame, padded[f*s.hop:f*s.hop+len(s.window)])
		for i, v := range frame {
			frame[i] = v * s.window[i]
		}
		re, im := s.fft.RealForward(frame)
		for k := range re {
			squared := re[k]*re[k] + im[k]*im[k]
			at := k*frames + f
			magnitude[at] = float32(math.Sqrt(float64(squared)))
			phase[at] = float32(math.Atan2(float64(im[k]), float64(re[k])))
		}
	}
	return magnitude, phase, frames
}

// Inverse rebuilds a waveform from a frequency-major magnitude and phase pair.
// numSamples is the length of the original input, which selects how much of the
// reference's window-multiple padding to trim back off.
func (s *STFT) Inverse(magnitude, phase []float32, numSamples int) []float32 {
	bins := s.fft.Bins()
	if len(magnitude)%bins != 0 || len(phase)%bins != 0 || len(magnitude) != len(phase) {
		panic("dsp: spectrum size mismatch")
	}
	frames := len(magnitude) / bins
	n, hop, width := len(s.window), s.hop, len(s.window)
	istft := hop * (frames - 1)
	accumulated, envelope := make([]float32, n+hop*(frames-1)), make([]float32, n+hop*(frames-1))
	re, im := make([]float32, bins), make([]float32, bins)
	for f := 0; f < frames; f++ {
		for k := 0; k < bins; k++ {
			at := k*frames + f
			angle := float64(phase[at])
			re[k], im[k] = magnitude[at]*float32(math.Cos(angle)), magnitude[at]*float32(math.Sin(angle))
		}
		block := s.fft.RealInverse(re, im)
		start := f * hop
		for i := 0; i < width; i++ {
			accumulated[start+i] += block[i] * s.window[i]
			envelope[start+i] += s.windowSq[i]
		}
	}
	// torch.istft divides by the accumulated window envelope, so a sample
	// covered by fewer window halves of use is scaled back up.
	out := make([]float32, istft)
	for i := range out {
		if denominator := envelope[i+s.center]; denominator > 1e-11 {
			out[i] = accumulated[i+s.center] / denominator
		}
	}
	// The reference trims its own window multiple; when the input already
	// divides the window, a whole window of zeros is removed.
	trim := n - numSamples%n
	if trim >= len(out) {
		return nil
	}
	return out[:len(out)-trim]
}

// pad appends enough zeros to reach a multiple of the window and then centres
// the result with the reflect padding torch.stft applies. The reference pads to
// a window multiple before torch.stft runs, so the reflect mirrors around the
// zero-extended signal, not around the original input. When the length already
// divides the window, a whole window of zeros is appended first, exactly as the
// reference's own pad expression does.
func (s *STFT) pad(x []float32) []float32 {
	target := padLength(len(x), len(s.window))
	base := make([]float32, target)
	copy(base, x)
	out := make([]float32, target+2*s.center)
	copy(out[s.center:], base)
	// Mirror to the left around the first sample of the zero-extended signal.
	for i := 0; i < s.center; i++ {
		out[i] = out[2*s.center-i]
	}
	// Mirror to the right around the last sample of the same signal.
	for i := 0; i < s.center; i++ {
		out[target+s.center+i] = out[target+s.center-2-i]
	}
	return out
}

// padLength returns the window multiple the reference pads to: the next
// multiple of window, or one window more when the input already divides it.
func padLength(length, window int) int {
	if remainder := length % window; remainder != 0 {
		return length + window - remainder
	}
	return length + window
}
