package irodori

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
)

func TestDACVAEDecoderParity(t *testing.T) {
	path := os.Getenv("IRODORI_CODEC_SAFE")
	if path == "" {
		t.Skip("set IRODORI_CODEC_SAFE for real decoder parity")
	}
	state, err := autograd.OpenSafeTensorFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	raw, err := os.ReadFile("../testdata/irodori_reference_latent.f32")
	if err != nil {
		t.Fatal(err)
	}
	latent := make([]float32, 4*32)
	for i := range latent {
		latent[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	waveBytes, err := os.ReadFile("../testdata/irodori_decoder_wave.f32")
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(waveBytes)/4)
	for i := range want {
		want[i] = math.Float32frombits(binary.LittleEndian.Uint32(waveBytes[i*4:]))
	}
	start := time.Now()
	got, err := (DACVAECodec{State: state}).Decode(latent)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("wave length %d want %d", len(got), len(want))
	}
	var maxAbs float64
	for i, v := range got {
		maxAbs = max(maxAbs, math.Abs(float64(v-want[i])))
	}
	t.Logf("wave_max_abs=%.9g go_ms=%.3f", maxAbs, float64(time.Since(start).Microseconds())/1000)
	if maxAbs > 0.002 {
		t.Errorf("decoder parity")
	}
}

func TestDACVAEReferenceEncodeParity(t *testing.T) {
	path := os.Getenv("IRODORI_CODEC_SAFE")
	root := os.Getenv("IRODORI_WAVS")
	if path == "" || root == "" {
		t.Skip("set IRODORI_CODEC_SAFE and IRODORI_WAVS for real codec parity")
	}
	state, err := autograd.OpenSafeTensorFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	if state.Metadata["format"] != "gograd-dacvae-f32-v1" {
		t.Fatal("wrong codec format")
	}
	b, err := os.ReadFile("../testdata/irodori_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Clips []struct {
			File   string `json:"file"`
			Latent struct {
				Shape     []int     `json:"shape"`
				First     []float32 `json:"first"`
				Last      []float32 `json:"last"`
				Sum       float64   `json:"sum"`
				SquareSum float64   `json:"square_sum"`
			} `json:"latent"`
			ElapsedMS float64 `json:"elapsed_ms"`
		} `json:"clips"`
	}
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, clip := range fixture.Clips {
		t.Run(clip.File, func(t *testing.T) {
			wav, sr, err := ReadWAVMono(filepath.Join(root, clip.File))
			if err != nil {
				t.Fatal(err)
			}
			if sr != 48000 {
				t.Skip("resampling pending")
			}
			start := time.Now()
			got, err := (DACVAECodec{State: state}).Encode48k(wav)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != clip.Latent.Shape[0]*32 {
				t.Fatalf("latent shape %d want %v", len(got), clip.Latent.Shape)
			}
			var maxAbs float64
			for i, want := range clip.Latent.First {
				maxAbs = max(maxAbs, math.Abs(float64(got[i]-want)))
			}
			for i, want := range clip.Latent.Last {
				maxAbs = max(maxAbs, math.Abs(float64(got[len(got)-len(clip.Latent.Last)+i]-want)))
			}
			var sum, sq float64
			for _, v := range got {
				sum += float64(v)
				sq += float64(v) * float64(v)
			}
			t.Logf("max_slice_abs=%.9g sum_abs=%.9g square_sum_abs=%.9g go_ms=%.3f torch_ms=%.3f", maxAbs, math.Abs(sum-clip.Latent.Sum), math.Abs(sq-clip.Latent.SquareSum), float64(time.Since(start).Microseconds())/1000, clip.ElapsedMS)
			if maxAbs > 0.001 {
				t.Errorf("codec encoder parity")
			}
		})
	}
}
