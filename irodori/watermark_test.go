package irodori

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/dsp"
)

type watermarkField struct {
	Field    string `json:"field"`
	Elements int    `json:"elements"`
}

type watermarkCase struct {
	N           int              `json:"n"`
	Rate        int              `json:"rate"`
	Seed        int              `json:"seed"`
	Offset      int64            `json:"offset"`
	Bytes       int              `json:"bytes"`
	OutLen      int              `json:"out_len"`
	Bins        int              `json:"bins"`
	Frames      int              `json:"frames"`
	Layout      []watermarkField `json:"layout"`
	Channels    []int            `json:"channels"`
	DeltaMaxAbs float64          `json:"delta_max_abs"`
}

type watermarkFixture struct {
	Cases []watermarkCase `json:"cases"`
}

// watermarkFields lists, in write order, how a case's blob region is split and
// which part of encodeParts each field is compared against. The encoded carrier
// is recorded for a spread of channels rather than all 32, which keeps the
// fixture small while still exercising every convolution and normalization.
var watermarkFields = []struct {
	name   string
	offset func(c watermarkCase) int
	stage  func(s watermarkStages, c watermarkCase, channels []int) []float32
}{
	{"audio", func(c watermarkCase) int { return c.N }, nil},
	{"out", func(c watermarkCase) int { return c.OutLen }, nil},
	{"carrier", func(c watermarkCase) int { return c.Bins * c.Frames },
		func(s watermarkStages, _ watermarkCase, _ []int) []float32 { return s.carrier }},
	{"rebuilt", func(c watermarkCase) int { return c.Bins * c.Frames },
		func(s watermarkStages, _ watermarkCase, _ []int) []float32 { return s.rebuilt }},
	{"sampled", func(c watermarkCase) int { return len(c.Channels) * c.Bins * c.Frames },
		func(s watermarkStages, c watermarkCase, _ []int) []float32 {
			out := make([]float32, len(c.Channels)*c.Bins*c.Frames)
			block := c.Bins * c.Frames
			for i, channel := range c.Channels {
				copy(out[i*block:], s.encoded[channel*block:(channel+1)*block])
			}
			return out
		}},
	{"raw", func(c watermarkCase) int { return c.Bins * c.Frames },
		func(s watermarkStages, _ watermarkCase, _ []int) []float32 { return s.perturbation }},
}

// loadWatermarkFixture reads the reference's own encode_wav pass. Each case packs
// its fields in the order watermarkFields lists them.
func loadWatermarkFixture(t *testing.T) ([]watermarkCase, []byte) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/irodori_watermark.json")
	if err != nil {
		t.Skip(err)
	}
	var f watermarkFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile("../testdata/irodori_watermark.bin")
	if err != nil {
		t.Fatal(err)
	}
	return f.Cases, blob
}

func openWatermark(t *testing.T) *SilentCipher {
	t.Helper()
	path := os.Getenv("IRODORI_WATERMARK_SAFE")
	if path == "" {
		t.Skip("set IRODORI_WATERMARK_SAFE to the converted watermark checkpoint")
	}
	state, err := autograd.OpenSafeTensorFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { state.Close() })
	model, err := LoadSilentCipher(state)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

// fieldBlob resolves a case's field to its byte range, checking the recorded
// layout so a misread cannot silently compare the wrong tensors.
func fieldBlob(t *testing.T, c watermarkCase, blob []byte) map[string][]float32 {
	t.Helper()
	out := make(map[string][]float32, len(c.Layout))
	at := c.Offset
	for _, field := range c.Layout {
		if len(field.Field) == 0 {
			t.Fatalf("case n=%d has an unnamed field", c.N)
		}
		out[field.Field] = readF32(blob, at, field.Elements)
		at += int64(field.Elements) * 4
	}
	if at != c.Offset+int64(c.Bytes) {
		t.Fatalf("case n=%d layout ends at %d, want %d", c.N, at, c.Offset+int64(c.Bytes))
	}
	return out
}

// TestWatermarkStages compares every intermediate grid so a divergence localizes
// to one stage. The carrier is only the STFT; the encoded carrier adds the
// encoder; the raw decoder output adds the carrier decoder and its
// post-processing; the rebuilt spectrum adds the perturbation arithmetic.
func TestWatermarkStages(t *testing.T) {
	cases, blob := loadWatermarkFixture(t)
	model := openWatermark(t)
	for _, c := range cases {
		audio := readF32(blob, c.Offset, c.N)
		fields := fieldBlob(t, c, blob)

		signal := audio
		if c.Rate != watermarkSampleRate {
			signal = dsp.Resample(audio, c.Rate, watermarkSampleRate)
		}
		stages := model.encodeParts(signal)
		if stages.frames != c.Frames {
			t.Fatalf("n=%d frames %d, want %d", c.N, stages.frames, c.Frames)
		}
		carrierAbs, at := maxAbs32(stages.carrier, fields["carrier"])
		carrierPeak := peakOf(stages.carrier)
		t.Logf("n=%d carrier max_abs=%.9g (peak %.9g)", c.N, carrierAbs, carrierPeak)
		if carrierAbs > 8*float64(carrierPeak)*1.2e-7+1e-4 {
			t.Errorf("n=%d carrier differs by %.9g at %d", c.N, carrierAbs, at)
		}
		for _, field := range watermarkFields[2:] {
			got := field.stage(stages, c, c.Channels)
			abs, where := maxAbs32(got, fields[field.name])
			t.Logf("n=%d %s max_abs=%.9g (at %d of %d)", c.N, field.name, abs, where, len(fields[field.name]))
			if abs > 1e-2 {
				t.Errorf("n=%d %s differs by %.9g at %d", c.N, field.name, abs, where)
			}
		}
	}
}

// TestWatermarkParity compares the waveform against the reference's own
// encode_wav. The bound sits well inside the watermark's own perturbation, which
// is what the model is designed to make dominant, and well above the port's
// float64-versus-float32 FFT error.
func TestWatermarkParity(t *testing.T) {
	cases, blob := loadWatermarkFixture(t)
	model := openWatermark(t)
	for _, c := range cases {
		audio := readF32(blob, c.Offset, c.N)
		wantOut := readF32(blob, c.Offset+int64(c.N)*4, c.OutLen)
		got, err := model.Encode(audio, c.Rate)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != c.OutLen {
			t.Errorf("n=%d rate=%d output length %d, want %d", c.N, c.Rate, len(got), c.OutLen)
			continue
		}
		abs, at := maxAbs32(got, wantOut)
		rel, _ := maxRel32(got, wantOut)
		t.Logf("n=%d rate=%d waveform max_abs=%.9g max_rel=%.9g (at %d, reference delta %.9g)",
			c.N, c.Rate, abs, rel, at, c.DeltaMaxAbs)
		if abs > c.DeltaMaxAbs/4+2e-3 {
			t.Errorf("n=%d rate=%d waveform differs by %.9g", c.N, c.Rate, abs)
		}
	}
}

// TestWatermarkMessageGrid checks the payload encoding against the reference's
// bit packing, symbol alphabet and 21-symbol tiling.
func TestWatermarkMessageGrid(t *testing.T) {
	grid, err := ExpandMessage([]byte(watermarkPayload), watermarkMessageDim, watermarkMessageLen, 47)
	if err != nil {
		t.Fatal(err)
	}
	if len(grid) != watermarkMessageDim*47 {
		t.Fatalf("grid has %d elements", len(grid))
	}
	for frame := 0; frame < 47; frame++ {
		var used int
		for k := 0; k < watermarkMessageDim; k++ {
			if grid[k*47+frame] != 0 {
				used++
			}
		}
		if used != 1 {
			t.Fatalf("frame %d selects %d symbols", frame, used)
		}
	}
	// 47 frames spans two full 21-symbol patterns plus a partial third, so the
	// 22nd and 43rd frames must repeat the first.
	for _, frame := range []int{21, 42} {
		for k := 0; k < watermarkMessageDim; k++ {
			if grid[k*47] != grid[k*47+frame] {
				t.Fatalf("symbol at frame %d does not repeat frame 0", frame)
			}
		}
	}
	if _, err := ExpandMessage([]byte("A"), watermarkMessageDim, watermarkMessageLen, 4); err == nil {
		t.Error("short payload accepted")
	}
}

// TestWatermarkSilence checks the reference's early return on silent input.
func TestWatermarkSilence(t *testing.T) {
	model := openWatermark(t)
	silent := make([]float32, 44100)
	out, err := model.Encode(silent, 44100)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(silent) {
		t.Fatalf("length %d, want %d", len(out), len(silent))
	}
	for i, v := range out {
		if v != 0 {
			t.Fatalf("silent input produced %v at %d", v, i)
		}
	}
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

func peakOf(v []float32) float32 {
	peak := float32(0)
	for _, x := range v {
		peak = max(peak, x)
	}
	return peak
}
