package irodori

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
)

func TestDiTCUDARealCheckpointParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME")
	}
	paths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "model.safetensors"))
	if err != nil || len(paths) != 1 {
		t.Fatal(err, paths)
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
		Output    [][]float32 `json:"output"`
		ElapsedMS float64     `json:"elapsed_ms"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	x := make([]float32, 4*32)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*.17) * .4)
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
	ctx, err := autograd.NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	start := time.Now()
	got, err := (CheckpointDiT{State: state, UseCUDA: true}).Forward(x, .7, text, []bool{true, true, true, false, false}, speaker, []bool{true, true}, caption, []bool{true, true, false})
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for row := range f.Output {
		for j, want := range f.Output[row] {
			maxAbs = max(maxAbs, math.Abs(float64(got[row*32+j]-want)))
		}
	}
	t.Logf("cuda_dit_max_abs=%.9g go_cuda_ms=%.3f torch_cpu_ms=%.3f", maxAbs, float64(time.Since(start).Microseconds())/1000, f.ElapsedMS)
	if maxAbs > 0.002 {
		t.Error("CUDA DiT parity")
	}
}
