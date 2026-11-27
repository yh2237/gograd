package dsp

import "math"

const maxResampleTableElements = 1 << 20

func resampleTap(phase, tap, width, orig, newRate int, base, scale float32) float32 {
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
	return sinc * window * scale
}

// Resample resamples a mono block with torchaudio's default Hann-windowed sinc
// polyphase filter.
func Resample(input []float32, from, to int) []float32 {
	if from <= 0 || to <= 0 || len(input) == 0 {
		return nil
	}
	if from == to {
		return append([]float32(nil), input...)
	}
	a, b := from, to
	for b != 0 {
		a, b = b, a%b
	}
	orig, newRate := from/a, to/a
	base := float32(min(orig, newRate)) * .99
	width := int(math.Ceil(6 * float64(orig) / float64(base)))
	kernelLength := 2*width + orig
	scale := base / float32(orig)
	outLength := int((int64(newRate)*int64(len(input)) + int64(orig) - 1) / int64(orig))
	out := make([]float32, outLength)

	if int64(newRate)*int64(kernelLength) <= maxResampleTableElements {
		kernels := make([]float32, newRate*kernelLength)
		for phase := 0; phase < newRate; phase++ {
			for tap := 0; tap < kernelLength; tap++ {
				kernels[phase*kernelLength+tap] = resampleTap(phase, tap, width, orig, newRate, base, scale)
			}
		}
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

	for q := range out {
		phase, step := q%newRate, q/newRate
		start := step*orig - width
		var sum float32
		lo := max(0, -start)
		hi := min(kernelLength, len(input)-start)
		for tap := lo; tap < hi; tap++ {
			sum += input[start+tap] * resampleTap(phase, tap, width, orig, newRate, base, scale)
		}
		out[q] = sum
	}
	return out
}
