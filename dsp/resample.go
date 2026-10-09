package dsp

import "math"

// Resample resamples a mono block from one rate to another with the kernel
// torchaudio.functional.resample builds by default: a Hann-windowed sinc
// polyphase filter using the greatest common divisor of the two rates.
//
// The rates are reduced by their common divisor first, exactly as the
// reference does, because the filter's base frequency and the number of phases
// both come from the reduced pair rather than the raw rates.
func Resample(input []float32, from, to int) []float32 {
	if from <= 0 || to <= 0 || len(input) == 0 {
		return nil
	}
	if from == to {
		return append([]float32(nil), input...)
	}
	gcd := func(a, b int) int {
		for b != 0 {
			a, b = b, a%b
		}
		return a
	}(from, to)
	orig, newRate := from/gcd, to/gcd
	base := float32(min(orig, newRate)) * .99
	width := int(math.Ceil(6 * float64(orig) / float64(base)))
	kernelLength := 2*width + orig
	kernels := make([]float32, newRate*kernelLength)
	for phase := 0; phase < newRate; phase++ {
		for tap := 0; tap < kernelLength; tap++ {
			idx := float32(tap-width) / float32(orig)
			t := (float32(-phase)/float32(newRate) + idx) * base
			t = max(-6, min(6, t))
			window := float32(math.Cos(float64(t * float32(math.Pi) / 12)))
			window *= window
			t *= float32(math.Pi)
			sinc := float32(1)
			if t != 0 {
				sinc = float32(math.Sin(float64(t))) / t
			}
			kernels[phase*kernelLength+tap] = sinc * window * (base / float32(orig))
		}
	}
	outLength := (newRate*len(input) + orig - 1) / orig
	out := make([]float32, outLength)
	for q := range out {
		phase, step := q%newRate, q/newRate
		start := step*orig - width
		var sum float32
		for tap := 0; tap < kernelLength; tap++ {
			at := start + tap
			if at >= 0 && at < len(input) {
				sum += input[at] * kernels[phase*kernelLength+tap]
			}
		}
		out[q] = sum
	}
	return out
}
