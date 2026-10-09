package irodori

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/torchrng"
)

// SamplerConfig describes the reference Euler rectified-flow schedule and CFG.
// The initial Gaussian noise is supplied by the caller, because a caller either
// injects the reference's own noise for a parity fixture or calls SampleSeeded,
// which draws it with the reference generator. Everything the reference's
// sampler exposes is settable; the three optional refinements default to off.
type SamplerConfig struct {
	Steps                                 int
	Schedule                              string // linear or sway
	SwayCoeff                             float32
	GuidanceMode                          string // independent, joint or alternating
	TextScale, SpeakerScale, CaptionScale float32
	CFGMinT, CFGMaxT                      float32
	UseCUDA                               bool // CUDA DiT projections, CPU attention/codec

	// TruncationFactor scales the initial noise when set, which the reference
	// applies before its first step.
	TruncationFactor float32

	// RescaleK and RescaleSigma enable temporal score rescaling and must be set
	// together, exactly as the reference requires.
	RescaleK, RescaleSigma float32

	// SpeakerKVScale multiplies the speaker context's keys and values, the
	// optional speaker inversion. A zero scale leaves the context untouched.
	// SpeakerKVLayers limits how many blocks are scaled; zero means all of them.
	SpeakerKVScale, SpeakerKVMinT float32
	SpeakerKVLayers               int
}

// validate checks the optional refinements against the reference's own rules.
func (cfg SamplerConfig) validate() error {
	if (cfg.RescaleK != 0) != (cfg.RescaleSigma != 0) {
		return fmt.Errorf("irodori: rescale_k and rescale_sigma must be set together")
	}
	if cfg.RescaleK != 0 && cfg.RescaleK <= 0 {
		return fmt.Errorf("irodori: rescale_k must be positive")
	}
	if cfg.RescaleSigma != 0 && cfg.RescaleSigma <= 0 {
		return fmt.Errorf("irodori: rescale_sigma must be positive")
	}
	if cfg.TruncationFactor < 0 {
		return fmt.Errorf("irodori: truncation_factor must not be negative")
	}
	if cfg.SpeakerKVScale != 0 && cfg.SpeakerKVScale < 0 {
		return fmt.Errorf("irodori: speaker_kv_scale must not be negative")
	}
	if cfg.SpeakerKVLayers < 0 {
		return fmt.Errorf("irodori: speaker_kv_max_layers must not be negative")
	}
	return nil
}

// speakerDeactivates reports whether a step crosses the speaker inversion's
// floor, after which the original magnitude is restored.
func (cfg SamplerConfig) speakerDeactivates(t, next float32) bool {
	return cfg.SpeakerKVMinT > 0 && next < cfg.SpeakerKVMinT && t >= cfg.SpeakerKVMinT
}

func (cfg SamplerConfig) times() ([]float32, error) {
	if cfg.Steps <= 0 {
		return nil, fmt.Errorf("irodori: sampler steps must be positive")
	}
	times := make([]float32, cfg.Steps+1)
	for j := range times {
		u := float32(j) / float32(cfg.Steps)
		switch cfg.Schedule {
		case "linear":
		case "sway":
			u += cfg.SwayCoeff * (float32(math.Cos(float64(float32(math.Pi/2)*u))) + u - 1)
			u = max(0, min(1, u))
		default:
			return nil, fmt.Errorf("irodori: unsupported schedule %q", cfg.Schedule)
		}
		times[j] = (1 - u) * .999
		if j > 0 && times[j] >= times[j-1] {
			return nil, fmt.Errorf("irodori: nondecreasing sampler schedule")
		}
	}
	return times, nil
}

func withoutCondition(c Conditions, which string) Conditions {
	switch which {
	case "text":
		c.Text = make([]float32, len(c.Text))
		c.TextMask = make([]bool, len(c.TextMask))
	case "speaker":
		c.Speaker = make([]float32, len(c.Speaker))
		c.SpeakerMask = make([]bool, len(c.SpeakerMask))
	case "caption":
		c.Caption = make([]float32, len(c.Caption))
		c.CaptionMask = make([]bool, len(c.CaptionMask))
	case "all":
		c = withoutCondition(withoutCondition(withoutCondition(c, "text"), "speaker"), "caption")
	}
	return c
}

// SampleEulerRF runs the reference timestep grid, CFG and Euler update with
// caller-supplied initial noise. It accepts conditions encoded once for all
// steps. The parity fixtures use this entry point because they inject the
// reference's own noise; SampleSeeded is the equivalent call that draws the
// noise with the reference generator instead.
func (c Conditioner) SampleEulerRF(noise []float32, condition Conditions, cfg SamplerConfig) ([]float32, error) {
	return c.sampleEulerRF(noise, condition, cfg)
}

// SampleSeeded runs the same sampler but draws the initial latent noise from
// the reference generator, so inference needs no Python-produced fixture. The
// reference seeds torch.Generator(device="cpu") with the same value, which
// torchrng reproduces bit for bit.
func (c Conditioner) SampleSeeded(condition Conditions, cfg SamplerConfig, frames int, seed uint64) ([]float32, error) {
	if frames <= 0 {
		return nil, fmt.Errorf("irodori: latent frames must be positive")
	}
	noise := make([]float32, frames*32)
	torchrng.New(seed).Normal(noise, 0, 1)
	return c.sampleEulerRF(noise, condition, cfg)
}

func (c Conditioner) sampleEulerRF(noise []float32, condition Conditions, cfg SamplerConfig) ([]float32, error) {
	if len(noise) == 0 || len(noise)%32 != 0 {
		return nil, fmt.Errorf("irodori: invalid initial noise")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	times, err := cfg.times()
	if err != nil {
		return nil, err
	}
	mode := cfg.GuidanceMode
	if mode != "independent" && mode != "joint" && mode != "alternating" {
		return nil, fmt.Errorf("irodori: unsupported CFG mode %q", mode)
	}
	x := append([]float32(nil), noise...)
	// The reference scales its initial noise by the truncation factor before the
	// first step, whatever produced the noise.
	if cfg.TruncationFactor != 0 && cfg.TruncationFactor != 1 {
		for i := range x {
			x[i] *= cfg.TruncationFactor
		}
	}
	names := make([]string, 0, 3)
	scales := map[string]float32{}
	for _, v := range []struct {
		name  string
		scale float32
	}{{"text", cfg.TextScale}, {"speaker", cfg.SpeakerScale}, {"caption", cfg.CaptionScale}} {
		if v.scale > 0 {
			names = append(names, v.name)
			scales[v.name] = v.scale
		}
	}
	if mode == "joint" && len(names) > 1 {
		for _, name := range names[1:] {
			if math.Abs(float64(scales[name]-scales[names[0]])) > 1e-6 {
				return nil, fmt.Errorf("irodori: joint CFG requires equal scales")
			}
		}
	}
	speakerActive := cfg.SpeakerKVScale != 0
	for step := 0; step < cfg.Steps; step++ {
		t, next := times[step], times[step+1]
		speaker := SpeakerContext{}
		if speakerActive && cfg.SpeakerKVScale != 0 {
			speaker = SpeakerContext{Scale: cfg.SpeakerKVScale, Layers: cfg.SpeakerKVLayers}
		}
		denoise := func(latent []float32, at float32, conditions Conditions, ctx SpeakerContext) ([]float32, error) {
			if cfg.UseCUDA {
				return c.DenoiseCUDA(latent, at, conditions, ctx)
			}
			return c.Denoise(latent, at, conditions, ctx)
		}
		velocity, err := denoise(x, t, condition, speaker)
		if err != nil {
			return nil, err
		}
		if len(names) > 0 && cfg.CFGMinT <= t && t <= cfg.CFGMaxT {
			selected := names
			if mode == "alternating" {
				selected = []string{names[step%len(names)]}
			}
			if mode == "joint" {
				selected = []string{"all"}
			}
			base := append([]float32(nil), velocity...)
			for _, name := range selected {
				// The reference scales every branch's speaker cache for independent
				// and alternating guidance, but leaves the joint unconditional branch
				// untouched.
				branchContext := speaker
				if name == "all" {
					branchContext = SpeakerContext{}
				}
				branch, err := denoise(x, t, withoutCondition(condition, name), branchContext)
				if err != nil {
					return nil, err
				}
				scale := scales[name]
				if name == "all" {
					scale = scales[names[0]]
				}
				for i := range velocity {
					velocity[i] += scale * (base[i] - branch[i])
				}
			}
		}
		if cfg.RescaleK != 0 && cfg.RescaleSigma != 0 {
			scaled, err := TemporalScoreRescale(velocity, x, t, cfg.RescaleK, cfg.RescaleSigma)
			if err != nil {
				return nil, err
			}
			velocity = scaled
		}
		for i := range x {
			x[i] += velocity[i] * (next - t)
		}
		if speakerActive && cfg.speakerDeactivates(t, next) {
			speakerActive = false
		}
	}
	return x, nil
}
