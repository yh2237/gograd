package dsp

import (
	"math"
	"testing"
)

// TestFFTMatchesNaiveDFT is the ground truth for the transform: a radix-2 FFT
// must agree with the definition to within float64 rounding.
func TestFFTMatchesNaiveDFT(t *testing.T) {
	for _, n := range []int{2, 4, 8, 16, 64, 1024, 4096} {
		f, err := NewFFT(n)
		if err != nil {
			t.Fatal(err)
		}
		re, im := make([]float64, n), make([]float64, n)
		for i := range re {
			re[i] = math.Sin(float64(i)*0.37) * 0.5
			im[i] = math.Cos(float64(i) * 0.11)
		}
		block := make([]float64, 2*n)
		for i := 0; i < n; i++ {
			block[2*i], block[2*i+1] = re[i], im[i]
		}
		f.Transform(block, false)

		var worst float64
		for k := 0; k < n; k++ {
			var sr, si float64
			for j := 0; j < n; j++ {
				angle := -2 * math.Pi * float64(k) * float64(j) / float64(n)
				c, s := math.Cos(angle), math.Sin(angle)
				sr += re[j]*c - im[j]*s
				si += re[j]*s + im[j]*c
			}
			d := math.Hypot(block[2*k]-sr, block[2*k+1]-si)
			scale := math.Hypot(sr, si) + 1
			worst = math.Max(worst, d/scale)
		}
		t.Logf("n=%d forward max relative error %.3g", n, worst)
		if worst > 1e-9 {
			t.Errorf("n=%d forward error %.3g", n, worst)
		}

		// Inverting what we just produced must return the input.
		restored := make([]float64, 2*n)
		copy(restored, block)
		f.Transform(restored, true)
		var back float64
		for i := 0; i < n; i++ {
			back = math.Max(back, math.Abs(restored[2*i]-re[i]))
			back = math.Max(back, math.Abs(restored[2*i+1]-im[i]))
		}
		if back > 1e-12 {
			t.Errorf("n=%d round trip error %.3g", n, back)
		}
	}
}

// TestRealForwardMatchesFullSpectrum checks the one-sided helpers against a
// manual complex transform of a real block.
func TestRealForwardMatchesFullSpectrum(t *testing.T) {
	f, err := NewFFT(8)
	if err != nil {
		t.Fatal(err)
	}
	block := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	re, im := f.RealForward(block)
	if len(re) != 5 || len(im) != 5 {
		t.Fatalf("bins %d, want 5", len(re))
	}
	full := make([]float64, 16)
	for i, v := range block {
		full[2*i] = float64(v)
	}
	f.Transform(full, false)
	for k := 0; k < 5; k++ {
		if math.Abs(float64(re[k])-full[2*k]) > 1e-5 || math.Abs(float64(im[k])-full[2*k+1]) > 1e-5 {
			t.Errorf("bin %d: got (%v,%v) want (%v,%v)", k, re[k], im[k], full[2*k], full[2*k+1])
		}
	}
	back := f.RealInverse(re, im)
	for i, v := range block {
		if math.Abs(float64(back[i])-float64(v)) > 1e-5 {
			t.Errorf("sample %d: got %v want %v", i, back[i], v)
		}
	}
}
