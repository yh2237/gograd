package autograd

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestPyTorchSafeTensorsRoundTrip(t *testing.T) {
	requirePyTorch(t)
	path := filepath.Join(t.TempDir(), "torch.safetensors")
	run := func(mode, path string) {
		t.Helper()
		cmd := exec.Command("python", "../tools/safetensors_fixture.py", mode, path)
		cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
		if out, e := cmd.CombinedOutput(); e != nil {
			t.Fatalf("%s: %v: %s", mode, e, out)
		}
	}
	run("write", path)
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, e := NewCUDAContext()
				if e != nil {
					t.Skip(e)
				}
				defer ctx.Close()
			}
			w, e := Zeros([]int{2, 2}, device, true)
			if e != nil {
				t.Fatal(e)
			}
			defer w.Close()
			b, e := Zeros([]int{2}, device, true)
			if e != nil {
				t.Fatal(e)
			}
			defer b.Close()
			m := &Module{Children: []NamedModule{{Name: "block", Module: &Module{Parameters: []Parameter{{Name: "weight", Value: w}, {Name: "bias", Value: b}}}}}}
			if e := m.LoadSafeTensors(path); e != nil {
				t.Fatal(e)
			}
			fixtureValues(t, "weight", w, []float32{1.25, -2.5, 3.75, 4.5}, 0)
			fixtureValues(t, "bias", b, []float32{-.125, .375}, 0)
			out := filepath.Join(t.TempDir(), "go.safetensors")
			if e := m.SaveSafeTensors(out); e != nil {
				t.Fatal(e)
			}
			run("verify", out)
		})
	}
}
