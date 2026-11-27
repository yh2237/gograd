package dsp

import (
	"math"
	"testing"
)

func TestResampleCoprimeRates(t *testing.T) {
	input := make([]float32, 64)
	for i := range input {
		input[i] = float32(math.Sin(2 * math.Pi * float64(i) / 16))
	}
	out := Resample(input, 44101, 48000)
	wantLen := int((int64(48000)*int64(len(input)) + 44101 - 1) / 44101)
	if len(out) != wantLen {
		t.Fatalf("len=%d want %d", len(out), wantLen)
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite value at %d: %v", i, v)
		}
	}
}
