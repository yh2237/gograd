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

func TestIntegratedCheckpointParity(t *testing.T) {
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME for real checkpoint parity")
	}
	base := filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*")
	paths, err := filepath.Glob(filepath.Join(base, "model.safetensors"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("checkpoint: %v %v", paths, err)
	}
	tokPaths, err := filepath.Glob(filepath.Join(base, "tokenizer", "tokenizer.json"))
	if err != nil || len(tokPaths) != 1 {
		t.Fatalf("tokenizer: %v %v", tokPaths, err)
	}
	state, err := autograd.OpenSafeTensorFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	tok, err := LoadTokenizer(tokPaths[0])
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../testdata/irodori_integrated.json")
	if err != nil {
		t.Fatal(err)
	}
	type fixtureTensor struct {
		Shape []int     `json:"shape"`
		First []float32 `json:"first"`
		Last  []float32 `json:"last"`
	}
	var f struct {
		Text         string        `json:"text"`
		Caption      string        `json:"caption"`
		TextState    fixtureTensor `json:"text_state"`
		SpeakerState fixtureTensor `json:"speaker_state"`
		CaptionState fixtureTensor `json:"caption_state"`
		Denoiser     fixtureTensor `json:"denoiser"`
		Duration     float32       `json:"duration"`
		ConditionMS  float64       `json:"condition_ms"`
		ForwardMS    float64       `json:"forward_ms"`
		DurationMS   float64       `json:"duration_ms"`
		Sampler      struct {
			Final     fixtureTensor `json:"final"`
			ElapsedMS float64       `json:"elapsed_ms"`
			Modes     map[string]struct {
				Final     fixtureTensor `json:"final"`
				ElapsedMS float64       `json:"elapsed_ms"`
			} `json:"modes"`
		} `json:"sampler"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/irodori_reference_latent.f32")
	if err != nil {
		t.Fatal(err)
	}
	ref := make([]float32, len(raw)/4)
	for i := range ref {
		ref[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	c := Conditioner{State: state, Tokenizer: tok}
	start := time.Now()
	cond, err := c.Encode(f.Text, f.Caption, ref, 24)
	if err != nil {
		t.Fatal(err)
	}
	encodeMS := float64(time.Since(start).Microseconds()) / 1000
	check := func(name string, got []float32, want fixtureTensor) {
		t.Helper()
		if len(got) == 0 || len(got) != product(want.Shape) {
			t.Fatalf("%s shape %d != %v", name, len(got), want.Shape)
		}
		var maxAbs float64
		for i, v := range want.First {
			maxAbs = max(maxAbs, math.Abs(float64(got[i]-v)))
		}
		for i, v := range want.Last {
			maxAbs = max(maxAbs, math.Abs(float64(got[len(got)-len(want.Last)+i]-v)))
		}
		t.Logf("%s max_slice_abs=%.9g", name, maxAbs)
		if maxAbs > 0.02 {
			t.Errorf("%s parity", name)
		}
	}
	check("text", cond.Text, f.TextState)
	check("caption", cond.Caption, f.CaptionState)
	check("speaker", cond.Speaker, f.SpeakerState)
	latent := make([]float32, 4*32)
	for i := range latent {
		latent[i] = float32(math.Sin(float64(i)*.17) * .4)
	}
	start = time.Now()
	velocity, err := c.Denoise(latent, .7, cond)
	if err != nil {
		t.Fatal(err)
	}
	forwardMS := float64(time.Since(start).Microseconds()) / 1000
	check("denoiser", velocity, f.Denoiser)
	start = time.Now()
	duration, err := c.PredictDuration(cond)
	if err != nil {
		t.Fatal(err)
	}
	durationMS := float64(time.Since(start).Microseconds()) / 1000
	t.Logf("duration_abs=%.9g go_condition_ms=%.3f torch_condition_ms=%.3f go_forward_ms=%.3f torch_forward_ms=%.3f go_duration_ms=%.3f torch_duration_ms=%.3f", math.Abs(float64(duration-f.Duration)), encodeMS, f.ConditionMS, forwardMS, f.ForwardMS, durationMS, f.DurationMS)
	if math.Abs(float64(duration-f.Duration)) > 0.02 {
		t.Error("duration parity")
	}
	noiseBytes, err := os.ReadFile("../testdata/irodori_sampler_noise.f32")
	if err != nil {
		t.Fatal(err)
	}
	noise := make([]float32, len(noiseBytes)/4)
	for i := range noise {
		noise[i] = math.Float32frombits(binary.LittleEndian.Uint32(noiseBytes[i*4:]))
	}
	start = time.Now()
	final, err := c.SampleEulerRF(noise, cond, SamplerConfig{Steps: 3, Schedule: "sway", SwayCoeff: -1, GuidanceMode: "independent", TextScale: 3, SpeakerScale: 5, CaptionScale: 3, CFGMinT: .5, CFGMaxT: 1})
	if err != nil {
		t.Fatal(err)
	}
	check("sampler", final, f.Sampler.Final)
	t.Logf("sampler go_ms=%.3f torch_ms=%.3f", float64(time.Since(start).Microseconds())/1000, f.Sampler.ElapsedMS)
	for _, mode := range []string{"joint", "alternating"} {
		fixture, ok := f.Sampler.Modes[mode]
		if !ok {
			t.Fatalf("missing %s fixture", mode)
		}
		speakerScale := float32(5)
		if mode == "joint" {
			speakerScale = 3
		}
		start = time.Now()
		result, err := c.SampleEulerRF(noise, cond, SamplerConfig{Steps: 3, Schedule: "sway", SwayCoeff: -1, GuidanceMode: mode, TextScale: 3, SpeakerScale: speakerScale, CaptionScale: 3, CFGMinT: .5, CFGMaxT: 1})
		if err != nil {
			t.Fatal(err)
		}
		check("sampler_"+mode, result, fixture.Final)
		t.Logf("sampler_%s go_ms=%.3f torch_ms=%.3f", mode, float64(time.Since(start).Microseconds())/1000, fixture.ElapsedMS)
	}
}

func product(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
