package irodori

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestTorchaudioReferenceResampleParity(t *testing.T) {
	read := func(path string) []float32 {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		v := make([]float32, len(b)/4)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return v
	}
	input := read("../testdata/irodori_resample_input.f32")
	want := read("../testdata/irodori_resample_output.f32")
	got, err := ResampleTo48k(input, 44100)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	var maxAbs float64
	for i, v := range got {
		maxAbs = max(maxAbs, math.Abs(float64(v-want[i])))
	}
	t.Logf("resample_max_abs=%.9g", maxAbs)
	if maxAbs > 0.0001 {
		t.Error("torchaudio resample parity")
	}
}

func TestReferenceWAVNormalizationParity(t *testing.T) {
	root := os.Getenv("IRODORI_WAVS")
	if root == "" {
		t.Skip("set IRODORI_WAVS for real clip parity")
	}
	b, err := os.ReadFile("../testdata/irodori_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Clips []struct {
			File       string `json:"file"`
			SampleRate int    `json:"sample_rate"`
			Samples    int    `json:"samples"`
			Normalized struct {
				First     []float32 `json:"first"`
				Last      []float32 `json:"last"`
				Middle    []float32 `json:"middle"`
				MaxAbs    float32   `json:"max_abs"`
				Sum       float32   `json:"sum"`
				SquareSum float32   `json:"square_sum"`
			} `json:"normalized"`
		} `json:"clips"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	for _, c := range f.Clips {
		t.Run(c.File, func(t *testing.T) {
			wav, sr, err := ReadWAVMono(filepath.Join(root, c.File))
			if err != nil {
				t.Fatal(err)
			}
			if sr != c.SampleRate || len(wav) != c.Samples {
				t.Fatalf("WAV shape %d/%d want %d/%d", sr, len(wav), c.SampleRate, c.Samples)
			}
			if sr != 48000 {
				t.Skip("resampling parity pending")
			}
			got, err := NormalizeReference48k(wav)
			if err != nil {
				t.Fatal(err)
			}
			var maxAbs float64
			for i, v := range c.Normalized.First {
				maxAbs = max(maxAbs, math.Abs(float64(got[i]-v)))
			}
			for i, v := range c.Normalized.Last {
				maxAbs = max(maxAbs, math.Abs(float64(got[len(got)-len(c.Normalized.Last)+i]-v)))
			}
			for i, v := range c.Normalized.Middle {
				maxAbs = max(maxAbs, math.Abs(float64(got[len(got)/2+i]-v)))
			}
			var peak float32
			for _, v := range got {
				peak = max(peak, float32(math.Abs(float64(v))))
			}
			var sum, sq float32
			for _, v := range got {
				sum += v
				sq += v * v
			}
			t.Logf("max_slice_abs=%.9g peak=%.9g torch_peak=%.9g sum_abs=%.9g square_sum_abs=%.9g", maxAbs, peak, c.Normalized.MaxAbs, math.Abs(float64(sum-c.Normalized.Sum)), math.Abs(float64(sq-c.Normalized.SquareSum)))
			if maxAbs > 0.0001 || math.Abs(float64(peak-c.Normalized.MaxAbs)) > 0.0001 {
				t.Errorf("normalized WAV mismatch")
			}
		})
	}
}
