package irodori

import (
	"encoding/binary"
	"encoding/json"
	"github.com/yh2237/gograd/autograd"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// integratedFixture loads the real checkpoint, the shared prompt conditions and
// the injected PyTorch noise once for the CFG audit tests.
type integratedFixture struct {
	Conditioner Conditioner
	Conditions  Conditions
	Noise       []float32
}

func openIntegratedFixture(t *testing.T) *integratedFixture {
	t.Helper()
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME")
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
	t.Cleanup(func() { state.Close() })
	tok, err := LoadTokenizer(tokPaths[0])
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../testdata/irodori_integrated.json")
	if err != nil {
		t.Fatal(err)
	}
	var prompt struct{ Text, Caption string }
	if err := json.Unmarshal(b, &prompt); err != nil {
		t.Fatal(err)
	}
	b, err = os.ReadFile("../testdata/irodori_reference_latent.f32")
	if err != nil {
		t.Fatal(err)
	}
	latent := littleEndianF32(b)
	c := Conditioner{State: state, Tokenizer: tok}
	cond, err := c.Encode(prompt.Text, prompt.Caption, latent, 24)
	if err != nil {
		t.Fatal(err)
	}
	b, err = os.ReadFile("../testdata/irodori_sampler_noise.f32")
	if err != nil {
		t.Fatal(err)
	}
	return &integratedFixture{Conditioner: c, Conditions: cond, Noise: littleEndianF32(b)}
}

func littleEndianF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// independentCFGBranches returns the reference conditional branch plus one
// guidance branch per conditioned context, in the reference's evaluation order.
func independentCFGBranches(t *testing.T, c Conditioner, x []float32, time float32, cond Conditions) [][]float32 {
	t.Helper()
	names := []string{"cond", "text", "speaker", "caption"}
	branches := make([][]float32, len(names))
	for branch, which := range names {
		condition := cond
		if branch > 0 {
			condition = withoutCondition(cond, which)
		}
		v, err := c.Denoise(x, time, condition)
		if err != nil {
			t.Fatal(err)
		}
		branches[branch] = v
	}
	return branches
}

// independentCFGVelocity composes the reference's independent guidance:
// conditional plus one scale*(conditional - drop-one-condition) term per branch.
// The reference applies guidance only inside its CFG time window.
func independentCFGVelocity(branches [][]float32, inWindow bool) []float32 {
	velocity := append([]float32(nil), branches[0]...)
	if !inWindow {
		return velocity
	}
	for branch, scale := range []float32{3, 5, 3} {
		for i := range velocity {
			velocity[i] += scale * (branches[0][i] - branches[branch+1][i])
		}
	}
	return velocity
}

// TestIndependentCFGBranchAudit checks each individual guidance branch at the
// first sampler step. The reference evaluates all four branches in one batched
// forward call, while gograd evaluates them serially, so the comparison is run
// against both PyTorch evaluations.
func TestIndependentCFGBranchAudit(t *testing.T) {
	fixture := openIntegratedFixture(t)
	b, err := os.ReadFile("../testdata/irodori_cfg_branches.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Batched [][][]float32 `json:"batched"`
		Serial  [][][]float32 `json:"serial"`
	}
	if err := json.Unmarshal(b, &reference); err != nil {
		t.Fatal(err)
	}
	for branch, which := range []string{"cond", "text", "speaker", "caption"} {
		got := independentCFGBranches(t, fixture.Conditioner, fixture.Noise, .999, fixture.Conditions)[branch]
		var maxBatch, maxSerial float64
		for row := range reference.Batched[branch] {
			for j, want := range reference.Batched[branch][row] {
				i := row*32 + j
				maxBatch = max(maxBatch, math.Abs(float64(got[i]-want)))
				maxSerial = max(maxSerial, math.Abs(float64(got[i]-reference.Serial[branch][row][j])))
			}
		}
		t.Logf("branch=%s go_vs_batch_max_abs=%.9g go_vs_serial_max_abs=%.9g", which, maxBatch, maxSerial)
	}
}

// modeTraceStep evaluates one reference sampler step: a conditional forward plus
// the guidance composition inside the CFG time window, followed by the Euler
// update. It mirrors the reference's own per-mode evaluation.
func modeTraceStep(t *testing.T, fixture *integratedFixture, x []float32, time, next float32, mode string, step int) []float32 {
	t.Helper()
	velocity, err := fixture.Conditioner.Denoise(x, time, fixture.Conditions)
	if err != nil {
		t.Fatal(err)
	}
	if .5 <= time && time <= 1 {
		names := []string{"text", "speaker", "caption"}
		scales := []float32{3, 5, 3}
		switch mode {
		case "independent":
			branches := [][]float32{velocity}
			for _, name := range names {
				branch, err := fixture.Conditioner.Denoise(x, time, withoutCondition(fixture.Conditions, name))
				if err != nil {
					t.Fatal(err)
				}
				branches = append(branches, branch)
			}
			velocity = independentCFGVelocity(branches, true)
		case "joint":
			all := withoutCondition(withoutCondition(withoutCondition(fixture.Conditions, "text"), "speaker"), "caption")
			uncond, err := fixture.Conditioner.Denoise(x, time, all)
			if err != nil {
				t.Fatal(err)
			}
			for i := range velocity {
				velocity[i] += scales[0] * (velocity[i] - uncond[i])
			}
		case "alternating":
			uncond, err := fixture.Conditioner.Denoise(x, time, withoutCondition(fixture.Conditions, names[step%len(names)]))
			if err != nil {
				t.Fatal(err)
			}
			scale := scales[step%len(scales)]
			for i := range velocity {
				velocity[i] += scale * (velocity[i] - uncond[i])
			}
		}
	}
	for i := range x {
		x[i] += velocity[i] * (next - time)
	}
	return velocity
}

// traceStep records one reference Euler step: the timestep, the next timestep,
// the composed velocity before the update, and the latent after the update.
type traceStep struct {
	T        float32     `json:"t"`
	Next     float32     `json:"next"`
	Velocity [][]float32 `json:"velocity"`
	X        [][]float32 `json:"x"`
}

// TestCFGTraceParity replays the reference's three-step sway Euler loop step by
// step for all three guidance modes. The reference reproduces the same
// trajectory, so per-step divergence isolates where the sampler and the
// reference differ and shows how each Euler step amplifies the residual
// float32 forward error.
func TestCFGTraceParity(t *testing.T) {
	fixture := openIntegratedFixture(t)
	b, err := os.ReadFile("../testdata/irodori_cfg_branches.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtureTrace struct {
		Traces map[string][]traceStep `json:"traces"`
	}
	if err := json.Unmarshal(b, &fixtureTrace); err != nil {
		t.Fatal(err)
	}
	if len(fixtureTrace.Traces) == 0 {
		t.Fatal("regenerate the audit fixture with IRODORI_CFG_AUDIT=1 tools/gen_irodori_integrated_fixture.py")
	}
	for _, mode := range []string{"independent", "joint", "alternating"} {
		x := append([]float32(nil), fixture.Noise...)
		var xIn, finalX float64
		for step := range fixtureTrace.Traces[mode] {
			want := fixtureTrace.Traces[mode][step]
			velocity := modeTraceStep(t, fixture, x, want.T, want.Next, mode, step)
			var maxVelocity, maxX float64
			for row := range want.Velocity {
				for j, v := range want.Velocity[row] {
					i := row*32 + j
					maxVelocity = max(maxVelocity, math.Abs(float64(velocity[i]-v)))
					maxX = max(maxX, math.Abs(float64(x[i]-want.X[row][j])))
				}
			}
			t.Logf("mode=%s step=%d t=%.9g x_in_max_abs=%.9g velocity_max_abs=%.9g x_max_abs=%.9g", mode, step, want.T, xIn, maxVelocity, maxX)
			xIn, finalX = maxX, maxX
		}
		// An algorithmic error already shows at O(1); the float32 noise floor
		// stays near 2e-3 for independent mode, so 0.02 leaves a wide margin.
		if finalX > 0.02 {
			t.Errorf("%s CFG trace parity: %.9g", mode, finalX)
		}
	}
}
