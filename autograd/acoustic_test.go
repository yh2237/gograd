package autograd

import (
	"encoding/json"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"os"
	"testing"
)

type fixtureParam struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type fixture struct {
	IDs     []int                   `json:"ids"`
	Speaker []int                   `json:"speaker"`
	Cont    []float32               `json:"cont"`
	Target  []float32               `json:"target"`
	Pred    []float32               `json:"pred"`
	Mask    []bool                  `json:"mask"`
	Params  map[string]fixtureParam `json:"params"`
	Grads   map[string][]float32    `json:"grads"`
	After   map[string][]float32    `json:"after"`
	Loss    float32                 `json:"loss"`
	Norm    float32                 `json:"norm"`
}

func loadFixture(t *testing.T) fixture {
	t.Helper()
	raw, e := os.ReadFile("../testdata/acoustic_step.json")
	if e != nil {
		t.Fatal(e)
	}
	var f fixture
	if e = json.Unmarshal(raw, &f); e != nil {
		t.Fatal(e)
	}
	return f
}
func closeValues(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d != %d", name, len(got), len(want))
	}
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > tol+tol*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d]: got %.9g want %.9g", name, i, got[i], want[i])
		}
	}
}
func testAcoustic(t *testing.T, device tensor.Device) {
	if device == tensor.CUDA {
		context, e := NewCUDAContext()
		if e != nil {
			t.Fatal(e)
		}
		defer context.Close()
		EnableGPUProfile(true)
		defer EnableGPUProfile(false)
	}
	f := loadFixture(t)
	m, e := NewAcoustic(4, 3, 5, []int{1, 2, 4, 8, 16, 1, 2, 4, 8, 16}, device)
	if e != nil {
		t.Fatal(e)
	}
	for _, p := range m.Params {
		ref, ok := f.Params[p.Name]
		if !ok {
			t.Fatalf("missing %s", p.Name)
		}
		if e := p.Value.CopyFrom(ref.Data); e != nil {
			t.Fatal(e)
		}
	}
	cont, e := New(f.Cont, []int{2, 3, 4}, device, false)
	if e != nil {
		t.Fatal(e)
	}
	target := append([]float32(nil), f.Target...)
	for i, ok := range f.Mask {
		if !ok {
			for j := 0; j < 5; j++ {
				target[i*5+j] = float32(math.NaN())
			}
		}
	}
	tgt, e := New(target, []int{2, 3, 5}, device, false)
	if e != nil {
		t.Fatal(e)
	}
	pred := m.Forward(f.IDs, cont, f.Speaker)
	if device == tensor.CUDA && (pred.Buffer() == nil || pred.Data != nil) {
		t.Fatal("CUDA output is not device-resident")
	}
	predValues, e := pred.ToHost()
	if e != nil {
		t.Fatal(e)
	}
	closeValues(t, "pred", predValues, f.Pred, 3e-4)
	loss := MaskedLoss(pred, tgt, false)
	lossValues, e := loss.ToHost()
	if e != nil {
		t.Fatal(e)
	}
	closeValues(t, "loss", lossValues, []float32{f.Loss}, 3e-4)
	if e = loss.Backward(); e != nil {
		t.Fatal(e)
	}
	for _, p := range m.Params {
		grad, e := p.Value.GradToHost()
		if e != nil {
			t.Fatal(e)
		}
		closeValues(t, "grad "+p.Name, grad, f.Grads[p.Name], 5e-4)
	}
	norm := ClipGradNorm(m.Params, 1)
	if device == tensor.CUDA {
		norm = GradientNormToHost()
	}
	closeValues(t, "norm", []float32{norm}, []float32{f.Norm}, 5e-4)
	opt := NewAdamW(m.Params, .001, .0001)
	opt.Step()
	for _, p := range m.Params {
		values, e := p.Value.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		closeValues(t, "step "+p.Name, values, f.After[p.Name], 5e-4)
	}
	if device == tensor.CUDA {
		profile := GPUProfile()
		for name, minCount := range map[string]int{"embedding_f": 2, "conv_forward_sgemm": 12, "group_f": 10, "binary_f": 10, "masked_f": 1, "global_norm_parts": 1, "scale_all_grads": 1} {
			if profile[name].Count < minCount {
				t.Fatalf("CUDA kernel %s launched %d times, need %d", name, profile[name].Count, minCount)
			}
		}
	}
}
func TestAcousticCPU(t *testing.T) { testAcoustic(t, tensor.CPU) }
func TestAcousticCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	testAcoustic(t, tensor.CUDA)
}
