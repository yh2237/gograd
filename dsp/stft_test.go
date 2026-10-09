package dsp

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
)

type stftCase struct {
	N             int     `json:"n"`
	Frames        int     `json:"frames"`
	Offset        int64   `json:"offset"`
	Bytes         int     `json:"bytes"`
	WavLen        int     `json:"wav_len"`
	ResampledLen  int     `json:"resampled_len"`
	RoundTripMaxA float64 `json:"round_trip_max_abs"`
}

type stftFixture struct {
	Cases []stftCase `json:"cases"`
	NFft  int        `json:"n_fft"`
	Hop   int        `json:"hop"`
}

// Each case packs: signal, magnitude, phase, inverse waveform, and the
// torchaudio-resampled signal, in that order and all little endian float32.
func loadSTFTFixture(t *testing.T) (stftFixture, []byte) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/dsp_stft.json")
	if err != nil {
		t.Skip(err)
	}
	var f stftFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile("../testdata/dsp_stft.bin")
	if err != nil {
		t.Fatal(err)
	}
	return f, blob
}

func readF32(b []byte, offset int64, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[offset+int64(i)*4:]))
	}
	return out
}

func maxAbs32(got, want []float32) (float64, int) {
	worst, at := 0.0, -1
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

func maxRel32(got, want []float32) (float64, int) {
	worst, at := 0.0, -1
	for i := range got {
		d := math.Abs(float64(got[i]-want[i])) / math.Max(math.Abs(float64(want[i])), 1e-30)
		if d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

// TestSTFTForward checks the magnitude and phase spectra against torch.stft,
// including the reference's epsilon guard and its window-multiple padding.
// The comparison is against the reconstructed spectrum the inverse consumes,
// because a phase difference of 2*pi at the arctangent's branch cut carries no
// information.
func TestSTFTForward(t *testing.T) {
	fixture, blob := loadSTFTFixture(t)
	s, err := NewSTFT(fixture.NFft, fixture.Hop)
	if err != nil {
		t.Fatal(err)
	}
	bins := s.Bins()
	for _, c := range fixture.Cases {
		signal := readF32(blob, c.Offset, c.N)
		wantMagnitude := readF32(blob, c.Offset+int64(c.N)*4, bins*c.Frames)
		wantPhase := readF32(blob, c.Offset+int64(c.N)*4+int64(bins*c.Frames)*4, bins*c.Frames)

		if got := s.FramesFor(c.N); got != c.Frames {
			t.Errorf("n=%d frames %d, want %d", c.N, got, c.Frames)
		}
		gotMagnitude, gotPhase, frames := s.Transform(signal)
		if frames != c.Frames {
			t.Errorf("n=%d transform frames %d, want %d", c.N, frames, c.Frames)
		}
		wantCos, wantSin := trigPair(wantMagnitude, wantPhase)
		gotCos, gotSin := trigPair(gotMagnitude, gotPhase)
		peak := peakOf(wantMagnitude)
		absCos, atCos := maxAbs32(gotCos, wantCos)
		absSin, _ := maxAbs32(gotSin, wantSin)
		absMag, atMag := maxAbs32(gotMagnitude, wantMagnitude)
		t.Logf("n=%d magnitude max_abs=%.9g spectrum max_abs=%.9g peak=%.9g (at %d/%d)",
			c.N, absMag, max(absCos, absSin), peak, atMag, atCos)
		// The reference evaluates the same transform in float32, so agreement
		// is expected to within a handful of float32 ulps of the peak bin.
		tolerance := 8*float64(peak)*1.2e-7 + 1e-6
		if float64(absMag) > tolerance {
			t.Errorf("n=%d magnitude differs by %.9g at %d", c.N, absMag, atMag)
		}
		if max(absCos, absSin) > tolerance {
			t.Errorf("n=%d spectrum differs by %.9g", c.N, max(absCos, absSin))
		}
	}
}

func trigPair(magnitude, phase []float32) (cos, sin []float32) {
	cos, sin = make([]float32, len(magnitude)), make([]float32, len(magnitude))
	for i := range magnitude {
		angle := float64(phase[i])
		cos[i] = magnitude[i] * float32(math.Cos(angle))
		sin[i] = magnitude[i] * float32(math.Sin(angle))
	}
	return cos, sin
}

func peakOf(magnitude []float32) float32 {
	peak := float32(0)
	for _, v := range magnitude {
		peak = max(peak, v)
	}
	return peak
}

// TestSTFTInverse checks the overlap-add inverse against torch.istft followed by
// the reference's window-multiple trim.
func TestSTFTInverse(t *testing.T) {
	fixture, blob := loadSTFTFixture(t)
	s, err := NewSTFT(fixture.NFft, fixture.Hop)
	if err != nil {
		t.Fatal(err)
	}
	bins := s.Bins()
	for _, c := range fixture.Cases {
		signal := readF32(blob, c.Offset, c.N)
		at := c.Offset + int64(c.N)*4
		wantMagnitude := readF32(blob, at, bins*c.Frames)
		at += int64(bins*c.Frames) * 4
		wantPhase := readF32(blob, at, bins*c.Frames)
		at += int64(bins*c.Frames) * 4
		wantWav := readF32(blob, at, c.WavLen)

		got := s.Inverse(wantMagnitude, wantPhase, c.N)
		if len(got) != c.WavLen {
			t.Errorf("n=%d inverse length %d, want %d", c.N, len(got), c.WavLen)
			continue
		}
		abs, at2 := maxAbs32(got, wantWav)
		t.Logf("n=%d inverse max_abs=%.9g (at %d) reference_round_trip=%.9g", c.N, abs, at2, c.RoundTripMaxA)
		if abs > 1e-5 {
			t.Errorf("n=%d inverse mismatch: max_abs=%.9g at %d", c.N, abs, at2)
		}
		// The residual must be no larger than the reference's own
		// transform-then-inverse error by more than a small factor.
		if abs > 20*c.RoundTripMaxA+1e-6 {
			t.Errorf("n=%d inverse residual %.9g exceeds reference round trip %.9g", c.N, abs, c.RoundTripMaxA)
		}
		// A direct round trip through both of our own directions must land
		// close to the original signal.
		magnitude, phase, frames := s.Transform(signal)
		if frames != c.Frames {
			t.Errorf("n=%d frames %d, want %d", c.N, frames, c.Frames)
		}
		rebuilt := s.Inverse(magnitude, phase, c.N)
		if len(rebuilt) != c.N {
			t.Errorf("n=%d rebuilt length %d, want %d", c.N, len(rebuilt), c.N)
			continue
		}
		abs, _ = maxAbs32(rebuilt, signal)
		t.Logf("n=%d go round trip max_abs=%.9g", c.N, abs)
		if abs > 20*c.RoundTripMaxA+1e-6 {
			t.Errorf("n=%d round trip %.9g exceeds reference %.9g", c.N, abs, c.RoundTripMaxA)
		}
	}
}

// TestResampleBothDirections checks the generalized polyphase resampler against
// torchaudio.functional.resample. The watermark needs the 48 kHz to 44.1 kHz
// direction, which the Irodori reference fixtures never exercised because they
// only resampled into the codec's rate.
func TestResampleBothDirections(t *testing.T) {
	fixture, blob := loadSTFTFixture(t)
	bins := 0
	for _, c := range fixture.Cases {
		signal := readF32(blob, c.Offset, c.N)
		// The fixture stores magnitude, phase and inverse waveform between the
		// signal and the resampled tail, so the tail starts after all of them.
		at := c.Offset + int64(c.N)*4 + int64(c.Frames*BinsFor(fixture.NFft))*4*2 + int64(c.WavLen)*4
		want := readF32(blob, at, c.ResampledLen)
		got := Resample(signal, 48000, 44100)
		if len(got) != c.ResampledLen {
			t.Fatalf("n=%d resampled length %d, want %d", c.N, len(got), c.ResampledLen)
		}
		abs, at2 := maxAbs32(got, want)
		rel, _ := maxRel32(got, want)
		t.Logf("n=%d resample 48k->44.1k max_abs=%.9g max_rel=%.9g (at %d) bins=%d", c.N, abs, rel, at2, bins)
		if abs > 1e-5 && rel > 1e-5 {
			t.Errorf("n=%d resample mismatch: max_abs=%.9g at %d", c.N, abs, at2)
		}
	}
}

// BinsFor reports the one-sided bin count for a filter length.
func BinsFor(filterLength int) int { return filterLength/2 + 1 }

func TestHannPeriodic(t *testing.T) {
	// torch.hann_window(4096): the window is symmetric about its centre and
	// starts at exactly zero for a periodic Hann.
	w := HannPeriodic(4096)
	if w[0] != 0 {
		t.Errorf("Hann[0] = %v, want 0", w[0])
	}
	if math.Abs(float64(w[2048])-1) > 1e-6 {
		t.Errorf("Hann[2048] = %v, want 1", w[2048])
	}
	if math.Abs(float64(w[1024])-0.5) > 1e-6 {
		t.Errorf("Hann[1024] = %v, want 0.5", w[1024])
	}
	if math.Abs(float64(w[2047])-1) > 1e-6 {
		t.Errorf("Hann[2047] = %v, want near 1", w[2047])
	}
}

func BenchmarkSTFTTransform(b *testing.B) {
	s, err := NewSTFT(4096, 2048)
	if err != nil {
		b.Fatal(err)
	}
	signal := make([]float32, 48000*3)
	for i := range signal {
		signal[i] = float32(math.Sin(float64(i) * 0.01))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Transform(signal)
	}
}
