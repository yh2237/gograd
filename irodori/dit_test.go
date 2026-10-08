package irodori

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
)

func TestDiTRealCheckpointParity(t *testing.T) {
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME for real checkpoint parity")
	}
	paths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "model.safetensors"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("checkpoint path: %v %v", paths, err)
	}
	state, err := autograd.OpenSafeTensorFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	b, err := os.ReadFile("../testdata/irodori_dit.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Output [][]float32 `json:"output"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	latent := make([]float32, 4*32)
	for i := range latent {
		latent[i] = float32(math.Sin(float64(i)*.17) * .4)
	}
	text := make([]float32, 5*512)
	for i := range text {
		text[i] = float32(math.Cos(float64(i)*.009) * .3)
	}
	speaker := make([]float32, 2*768)
	for i := range speaker {
		speaker[i] = float32(math.Sin(float64(i)*.006) * .2)
	}
	caption := make([]float32, 3*512)
	for i := range caption {
		caption[i] = float32(math.Cos(float64(i)*.012) * .25)
	}
	start := time.Now()
	got, err := (CheckpointDiT{State: state}).Forward(latent, .7, text, []bool{true, true, true, false, false}, speaker, []bool{true, true}, caption, []bool{true, true, false})
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for row := range f.Output {
		for j, want := range f.Output[row] {
			maxAbs = max(maxAbs, math.Abs(float64(got[row*32+j]-want)))
		}
	}
	t.Logf("max_abs=%.9g go_ms=%.3f", maxAbs, float64(time.Since(start).Microseconds())/1000)
	if maxAbs > 0.002 {
		t.Errorf("DiT forward mismatch")
	}
}
