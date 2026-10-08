package irodori

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
)

func TestRealCheckpointCUDAProjectionParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME for real checkpoint parity")
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
	w, shape, err := state.ReadF32("in_proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	if shape[0] != 1280 || shape[1] != 32 {
		t.Fatal(shape)
	}
	x := make([]float32, 4*32)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*.17) * .4)
	}
	want := linearRows(x, w, 4, 32, 1280)
	ctx, err := autograd.NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	start := time.Now()
	got, err := LinearRowsCUDA(x, w, 4, 32, 1280)
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i, v := range got {
		maxAbs = max(maxAbs, math.Abs(float64(v-want[i])))
	}
	t.Logf("real_checkpoint_cuda_projection_max_abs=%.9g cuda_ms=%.3f", maxAbs, float64(time.Since(start).Microseconds())/1000)
	start = time.Now()
	for i := 0; i < 20; i++ {
		if _, err := LinearRowsCUDA(x, w, 4, 32, 1280); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("real_checkpoint_cuda_projection_warm_avg_ms=%.3f", float64(time.Since(start).Microseconds())/20000)
	if maxAbs > 0.002 {
		t.Error("CUDA projection parity")
	}
}
