package dsp

import "testing"

func TestAuditFFTIdentity(t *testing.T) {
	f, err := NewFFT(1)
	if err != nil {
		t.Fatal(err)
	}
	if f.Len() != 1 || f.Bins() != 1 {
		t.Fatalf("unexpected identity FFT dimensions: %d, %d", f.Len(), f.Bins())
	}
	for _, inverse := range []bool{false, true} {
		x := []float64{3.25, -2.5}
		f.Transform(x, inverse)
		if x[0] != 3.25 || x[1] != -2.5 {
			t.Fatalf("inverse=%v: got %v", inverse, x)
		}
	}
	re, im := f.RealForward([]float32{-4.5})
	if len(re) != 1 || re[0] != -4.5 || im[0] != 0 {
		t.Fatalf("real forward: %v + i%v", re, im)
	}
	if got := f.RealInverse(re, im); len(got) != 1 || got[0] != -4.5 {
		t.Fatalf("real inverse: %v", got)
	}
}
