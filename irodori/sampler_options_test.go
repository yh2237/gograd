package irodori

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
)

type samplerOptionCase struct {
	Name   string `json:"name"`
	Values int    `json:"values"`
	Offset int64  `json:"offset"`
}

type samplerOptionFixture struct {
	Cases []samplerOptionCase `json:"cases"`
}

// samplerOptionConfigs mirrors the generator's cases. Each entry is the
// SamplerConfig the reference is run with, in the same order.
var samplerOptionConfigs = map[string]SamplerConfig{
	"baseline":          {},
	"truncation":        {TruncationFactor: 0.85},
	"rescale":           {RescaleK: 1.2, RescaleSigma: 0.7},
	"speaker_kv":        {SpeakerKVScale: 1.5},
	"speaker_kv_floor":  {SpeakerKVScale: 1.5, SpeakerKVMinT: 0.8},
	"speaker_kv_layers": {SpeakerKVScale: 1.25, SpeakerKVLayers: 4},
	"all": {TruncationFactor: 0.9, RescaleK: 1.2, RescaleSigma: 0.7,
		SpeakerKVScale: 1.5, SpeakerKVMinT: 0.8},
}

// TestSamplerOptionParity checks the optional refinements against the
// reference's own loop. The tolerance is the same one the sampled-latent
// comparisons use, because the reference batches its CFG branches while gograd
// evaluates them serially and the two differ by float32 rounding.

// readReferenceNoise loads one of the Python-seeded noise fixtures the parity
// tests inject, which keeps every sampler comparison on the same input.
func readReferenceNoise(t *testing.T, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile("../testdata/" + name)
	if err != nil {
		t.Skip(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func TestSamplerOptionParity(t *testing.T) {
	fixture := openIntegratedFixture(t)
	raw, err := os.ReadFile("../testdata/irodori_sampler_options.json")
	if err != nil {
		t.Skip(err)
	}
	var index samplerOptionFixture
	if err := json.Unmarshal(raw, &index); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile("../testdata/irodori_sampler_options.bin")
	if err != nil {
		t.Fatal(err)
	}
	noise := readReferenceNoise(t, "irodori_sampler_noise.f32")
	for _, c := range index.Cases {
		override, ok := samplerOptionConfigs[c.Name]
		if !ok {
			t.Fatalf("no config recorded for %q", c.Name)
		}
		want := make([]float32, c.Values)
		for i := range want {
			want[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[c.Offset+int64(i)*4:]))
		}
		cfg := SamplerConfig{Steps: 3, Schedule: "sway", SwayCoeff: -1,
			GuidanceMode: "independent", TextScale: 3, SpeakerScale: 5,
			CaptionScale: 3, CFGMinT: .5, CFGMaxT: 1}
		cfg.TruncationFactor = override.TruncationFactor
		cfg.RescaleK, cfg.RescaleSigma = override.RescaleK, override.RescaleSigma
		cfg.SpeakerKVScale, cfg.SpeakerKVMinT = override.SpeakerKVScale, override.SpeakerKVMinT
		cfg.SpeakerKVLayers = override.SpeakerKVLayers
		got, err := fixture.Conditioner.SampleEulerRF(noise, fixture.Conditions, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != c.Values {
			t.Fatalf("%s length %d, want %d", c.Name, len(got), c.Values)
		}
		abs, at := maxAbs32(got, want)
		t.Logf("%s final latent max_abs=%.9g (at %d)", c.Name, abs, at)
		if abs > 0.02 {
			t.Errorf("%s final latent differs by %.9g at %d", c.Name, abs, at)
		}
	}
}

// TestSamplerOptionValidation checks that the paired and signed options reject
// the combinations the reference rejects.
func TestSamplerOptionValidation(t *testing.T) {
	for name, cfg := range map[string]SamplerConfig{
		"rescale without sigma":   {RescaleK: 1.2},
		"rescale without k":       {RescaleSigma: 0.7},
		"negative rescale k":      {RescaleK: -1, RescaleSigma: 0.7},
		"negative truncation":     {TruncationFactor: -0.5},
		"negative speaker scale":  {SpeakerKVScale: -1},
		"negative speaker layers": {SpeakerKVScale: 1, SpeakerKVLayers: -1},
	} {
		if err := cfg.validate(); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
	for name, cfg := range map[string]SamplerConfig{
		"no options":         {},
		"rescale pair":       {RescaleK: 1.2, RescaleSigma: 0.7},
		"speaker scale":      {SpeakerKVScale: 1.5},
		"truncation":         {TruncationFactor: 0.85},
		"all speaker bounds": {SpeakerKVScale: 1.5, SpeakerKVMinT: 0.8, SpeakerKVLayers: 3},
	} {
		if err := cfg.validate(); err != nil {
			t.Errorf("%s was rejected: %v", name, err)
		}
	}
}

// TestSpeakerDeactivation checks the floor rule: the reference restores the
// speaker magnitude when a step crosses the floor downward, and never applies a
// factor when no floor is configured.
func TestSpeakerDeactivation(t *testing.T) {
	if cfg := (SamplerConfig{SpeakerKVScale: 1.5}); cfg.speakerDeactivates(0.9, 0.5) {
		t.Error("no floor configured but deactivated")
	}
	cfg := SamplerConfig{SpeakerKVScale: 1.5, SpeakerKVMinT: 0.8}
	if !cfg.speakerDeactivates(0.9, 0.5) {
		t.Error("crossing the floor did not deactivate")
	}
	if cfg.speakerDeactivates(0.7, 0.5) {
		t.Error("already below the floor but deactivated again")
	}
	if cfg.speakerDeactivates(0.9, 0.85) {
		t.Error("staying above the floor deactivated")
	}
}

// TestTemporalScoreRescaleAtOne checks the reference's early return at t = 1,
// where the correction is a no-op.
func TestTemporalScoreRescaleAtOne(t *testing.T) {
	velocity := []float32{1, 2, 3}
	latent := []float32{4, 5, 6}
	out, err := TemporalScoreRescale(velocity, latent, 1, 1.2, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	for i := range out {
		if out[i] != velocity[i] {
			t.Fatalf("t=1 changed element %d to %v", i, out[i])
		}
	}
}
