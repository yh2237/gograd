package irodori

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
)

func TestResidentDenoiserBenchmark(t *testing.T) {
	if os.Getenv("IRODORI_RESIDENT_BENCH") == "" {
		t.Skip("opt in with IRODORI_RESIDENT_BENCH=1 (about 1.5 GB host RAM)")
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
	p := CheckpointDiT{State: state}
	forward := func() ([]float32, error) {
		return p.Forward(x, .7, text, []bool{true, true, true, false, false}, speaker, []bool{true, true}, caption, []bool{true, true, false})
	}
	start := time.Now()
	streamed, err := forward()
	if err != nil {
		t.Fatal(err)
	}
	streamMS := float64(time.Since(start).Microseconds()) / 1000
	start = time.Now()
	if err := PreloadDenoiser(state); err != nil {
		t.Fatal(err)
	}
	preloadMS := float64(time.Since(start).Microseconds()) / 1000
	start = time.Now()
	resident, err := forward()
	if err != nil {
		t.Fatal(err)
	}
	residentMS := float64(time.Since(start).Microseconds()) / 1000
	var maxAbs float64
	for i, v := range streamed {
		maxAbs = max(maxAbs, math.Abs(float64(v-resident[i])))
	}
	t.Logf("cached_bytes=%d preload_ms=%.3f streamed_ms=%.3f resident_ms=%.3f max_abs=%.9g", state.CachedBytes(), preloadMS, streamMS, residentMS, maxAbs)
	if maxAbs > 0 {
		t.Error("resident output changed")
	}
}
