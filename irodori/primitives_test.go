package irodori

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type primitiveFixture struct {
	RMSInput         []float32            `json:"rms_input"`
	RMSWeight        []float32            `json:"rms_weight"`
	RMSOutput        []float32            `json:"rms_output"`
	RotaryInput      []float32            `json:"rotary_input"`
	RotaryOutput     []float32            `json:"rotary_output"`
	TimestepInput    []float32            `json:"timestep_input"`
	TimestepOutput   []float32            `json:"timestep_output"`
	ScheduleLinear   []float32            `json:"schedule_linear"`
	ScheduleSway     []float32            `json:"schedule_sway"`
	RescaleVelocity  []float32            `json:"rescale_velocity"`
	RescaleLatent    []float32            `json:"rescale_latent"`
	RescaleOutput    []float32            `json:"rescale_output"`
	DurationTexts    []string             `json:"duration_texts"`
	DurationTokens   []int                `json:"duration_tokens"`
	DurationSpeakers []bool               `json:"duration_speakers"`
	DurationOutputs  [][]float32          `json:"duration_outputs"`
	AdaLNInput       []float32            `json:"adaln_input"`
	AdaLNCond        []float32            `json:"adaln_cond"`
	AdaLNOutput      []float32            `json:"adaln_output"`
	AdaLNGate        []float32            `json:"adaln_gate"`
	AdaLNState       map[string][]float32 `json:"adaln_state"`
}

func loadFixture(b testing.TB) primitiveFixture {
	b.Helper()
	data, err := os.ReadFile("../testdata/irodori_primitives.json")
	if err != nil {
		b.Fatal(err)
	}
	var f primitiveFixture
	if err := json.Unmarshal(data, &f); err != nil {
		b.Fatal(err)
	}
	return f
}

func compareFixture(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	var maxAbs, maxRel float64
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		maxAbs = max(maxAbs, diff)
		maxRel = max(maxRel, diff/math.Max(math.Abs(float64(want[i])), 1e-6))
	}
	t.Logf("%s: max_abs=%.9g max_rel=%.9g", name, maxAbs, maxRel)
	if maxAbs > tolerance {
		t.Errorf("%s: max abs %.9g exceeds %.9g", name, maxAbs, tolerance)
	}
}

func TestPyTorchPrimitiveParity(t *testing.T) {
	f := loadFixture(t)
	rms, err := RMSNorm(f.RMSInput, f.RMSWeight, 4, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "RMSNorm", rms, f.RMSOutput, 2e-6)
	rope, err := Rotary(f.RotaryInput, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "RoPE", rope, f.RotaryOutput, 3e-6)
	time, err := TimestepEmbedding(f.TimestepInput, 8)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "timestep", time, f.TimestepOutput, 2e-5)
	linear, err := RFSchedule(8, "linear", -1)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "linear schedule", linear, f.ScheduleLinear, 1e-7)
	sway, err := RFSchedule(8, "sway", -1)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "sway schedule", sway, f.ScheduleSway, 2e-7)
	rescale, err := TemporalScoreRescale(f.RescaleVelocity, f.RescaleLatent, .6, 1.2, .7)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "score rescale", rescale, f.RescaleOutput, 3e-7)
	for i, s := range f.DurationTexts {
		features, err := DurationFeatures(s, f.DurationTokens[i], 256, f.DurationSpeakers[i])
		if err != nil {
			t.Fatal(err)
		}
		compareFixture(t, "duration features", features[:], f.DurationOutputs[i], 2e-7)
	}
	projection := func(prefix string) LowRankProjection {
		return LowRankProjection{
			Down: f.AdaLNState[prefix+"_down.weight"],
			Up:   f.AdaLNState[prefix+"_up.weight"],
			Bias: f.AdaLNState[prefix+"_up.bias"],
		}
	}
	ada := LowRankAdaLN{Width: 8, Rank: 3, Epsilon: 1e-5,
		Shift: projection("shift"), Scale: projection("scale"), Gate: projection("gate")}
	modulated, gates, err := ada.Forward(f.AdaLNInput, f.AdaLNCond)
	if err != nil {
		t.Fatal(err)
	}
	compareFixture(t, "LowRankAdaLN output", modulated, f.AdaLNOutput, 5e-7)
	compareFixture(t, "LowRankAdaLN gate", gates, f.AdaLNGate, 5e-7)
}

func BenchmarkRMSNorm(b *testing.B) {
	input := make([]float32, 256*1024)
	weight := make([]float32, 1024)
	for i := range weight {
		weight[i] = 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSNorm(input, weight, 1024, 1e-6); err != nil {
			b.Fatal(err)
		}
	}
}
