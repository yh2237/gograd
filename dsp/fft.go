// Package dsp holds the signal-processing primitives the Irodori reference
// needs but that are not specific to it: a power-of-two complex FFT, the
// periodic Hann window, the magnitude/phase STFT the watermark works in, and a
// polyphase resampler.
//
// All of these mirror PyTorch and torchaudio behaviour rather than a textbook
// ideal, because they exist to reproduce a reference pipeline bit for bit. The
// FFT accumulates in float64 and rounds on the way out: PyTorch's float32
// pocketfft differs from that by about one ulp, which is far below the noise
// floor of the models that consume the spectra.
package dsp

import "math"

// FFT is an iterative radix-2 complex transform of a fixed power-of-two
// length. The twiddle table is built once, so a transform only costs butterflies.
type FFT struct {
	n    int
	cos  []float64
	sin  []float64
	rev  []int
	log2 int
}

// NewFFT returns a transform for blocks of n samples, which must be a power of
// two.
func NewFFT(n int) (*FFT, error) {
	if n <= 0 || n&(n-1) != 0 {
		return nil, errNotPowerOfTwo(n)
	}
	log2 := 0
	for m := n; m > 1; m >>= 1 {
		log2++
	}
	f := &FFT{n: n, cos: make([]float64, n/2), sin: make([]float64, n/2), rev: make([]int, n), log2: log2}
	for k := range f.cos {
		angle := 2 * math.Pi * float64(k) / float64(n)
		f.cos[k], f.sin[k] = math.Cos(angle), math.Sin(angle)
	}
	for i := range f.rev {
		f.rev[i] = f.rev[i>>1]>>1 | (i&1)<<(log2-1)
	}
	return f, nil
}

// Len reports the transform length.
func (f *FFT) Len() int { return f.n }

// Transform replaces x, stored as interleaved real/imaginary pairs, with its
// discrete transform. The inverse divides by n, as a discrete inverse must.
func (f *FFT) Transform(x []float64, inverse bool) {
	if len(x) != 2*f.n {
		panic("dsp: FFT block length mismatch")
	}
	f.bitReverse(x)
	if inverse {
		// x = conj(DFT(conj(X)))/n, which is the inverse transform because
		// conjugating first turns the forward transform into a backward one.
		for i := 0; i < len(x); i += 2 {
			x[i+1] = -x[i+1]
		}
		f.butterflies(x, -1)
		scale := 1 / float64(f.n)
		for i := 0; i < len(x); i += 2 {
			x[i], x[i+1] = x[i]*scale, -x[i+1]*scale
		}
		return
	}
	f.butterflies(x, -1)
}

func (f *FFT) butterflies(x []float64, twiddleSign float64) {
	for size := 2; size <= f.n; size <<= 1 {
		half, step := size/2, f.n/size
		for start := 0; start < f.n; start += size {
			for k := 0; k < half; k++ {
				twiddle := k * step
				wr, wi := f.cos[twiddle], twiddleSign*f.sin[twiddle]
				i, j := 2*(start+k), 2*(start+k+half)
				br, bi := x[j], x[j+1]
				tr, ti := br*wr-bi*wi, br*wi+bi*wr
				x[j] = x[i] - tr
				x[j+1] = x[i+1] - ti
				x[i], x[i+1] = x[i]+tr, x[i+1]+ti
			}
		}
	}
}

func (f *FFT) bitReverse(x []float64) {
	for i := range f.rev {
		if j := f.rev[i]; i < j {
			a, b := 2*i, 2*j
			x[a], x[b] = x[b], x[a]
			x[a+1], x[b+1] = x[b+1], x[a+1]
		}
	}
}

// Bins reports the number of one-sided bins, n/2+1.
func (f *FFT) Bins() int { return f.n/2 + 1 }

// RealForward transforms a real block of n samples and returns the one-sided
// spectrum as float32 pairs, matching torch.stft's onesided output.
func (f *FFT) RealForward(block []float32) (re, im []float32) {
	if len(block) != f.n {
		panic("dsp: FFT block length mismatch")
	}
	buf := make([]float64, 2*f.n)
	for i, v := range block {
		buf[2*i] = float64(v)
	}
	f.bitReverse(buf)
	f.butterflies(buf, -1)
	re, im = make([]float32, f.n/2+1), make([]float32, f.n/2+1)
	for k := range re {
		re[k], im[k] = float32(buf[2*k]), float32(buf[2*k+1])
	}
	return re, im
}

// RealInverse rebuilds a real block of n samples from the one-sided spectrum,
// mirroring it into the missing half before transforming.
func (f *FFT) RealInverse(re, im []float32) []float32 {
	if len(re) != f.n/2+1 || len(im) != f.n/2+1 {
		panic("dsp: FFT bin count mismatch")
	}
	buf := make([]float64, 2*f.n)
	for k := range re {
		buf[2*k], buf[2*k+1] = float64(re[k]), float64(im[k])
	}
	for k := 1; k < f.n/2; k++ {
		buf[2*(f.n-k)] = float64(re[k])
		buf[2*(f.n-k)+1] = -float64(im[k])
	}
	f.Transform(buf, true)
	out := make([]float32, f.n)
	for i := range out {
		out[i] = float32(buf[2*i])
	}
	return out
}
