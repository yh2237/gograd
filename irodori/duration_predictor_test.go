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

func TestDurationRealCheckpointParity(t *testing.T) {
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
	b, err := os.ReadFile("../testdata/irodori_duration.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			HasSpeaker bool    `json:"has_speaker"`
			HasCaption bool    `json:"has_caption"`
			LogFrames  float32 `json:"log_frames"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	text := make([]float32, 5*512)
	for i := range text {
		text[i] = float32(math.Sin(float64(i) * .013))
	}
	speaker := make([]float32, 2*768)
	for i := range speaker {
		speaker[i] = float32(math.Cos(float64(i) * .007))
	}
	caption := make([]float32, 3*512)
	for i := range caption {
		caption[i] = float32(math.Sin(float64(i) * .011))
	}
	for _, c := range f.Cases {
		start := time.Now()
		got, err := (CheckpointDurationPredictor{State: state}).Forward(text, []bool{true, true, true, false, false}, speaker, caption, []bool{true, true, false}, c.HasSpeaker, c.HasCaption)
		if err != nil {
			t.Fatal(err)
		}
		diff := math.Abs(float64(got - c.LogFrames))
		t.Logf("speaker=%v caption=%v max_abs=%.9g go_ms=%.3f", c.HasSpeaker, c.HasCaption, diff, float64(time.Since(start).Microseconds())/1000)
		if diff > 2e-4 {
			t.Errorf("duration mismatch: got %.9g want %.9g", got, c.LogFrames)
		}
	}
}
