package autograd

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type fixtureTensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
	Grad  []float32 `json:"grad"`
}
type fixtureCase struct {
	X, W, B, A, Q, K, V, Mask, Up, Y fixtureTensor
	Target, IDs                      []int
	Params                           map[string]fixtureTensor
	Grads, Step                      map[string][]float32
}

func loadTransformerFixture(t *testing.T, device tensor.Device) map[string]fixtureCase {
	t.Helper()
	requirePyTorch(t)
	path := filepath.Join(t.TempDir(), "fixture.json")
	arg := "cpu"
	if device == tensor.CUDA {
		arg = "cuda"
	}
	cmd := exec.Command("python", "../tools/gen_transformer_fixture.py", path, arg)
	cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("PyTorch fixture: %v: %s", err, out)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var f map[string]fixtureCase
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f
}
func fixtureNew(t *testing.T, f fixtureTensor, device tensor.Device, grad bool) *Tensor {
	t.Helper()
	x, e := New(f.Data, f.Shape, device, grad)
	if e != nil {
		t.Fatal(e)
	}
	return x
}
func fixtureClose(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d != %d", label, len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > tol+tol*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d]: got %.8g, want %.8g", label, i, got[i], want[i])
		}
	}
}
func fixtureValues(t *testing.T, label string, x *Tensor, want []float32, tol float64) {
	t.Helper()
	v, e := x.ToHost()
	if e != nil {
		t.Fatal(e)
	}
	fixtureClose(t, label, v, want, tol)
}
func fixtureGrad(t *testing.T, label string, x *Tensor, want []float32, tol float64) {
	t.Helper()
	v, e := x.GradToHost()
	if e != nil {
		t.Fatal(e)
	}
	fixtureClose(t, label, v, want, tol)
}
func weightedSum(y, up *Tensor) *Tensor {
	z := Mul(y, up)
	axes := make([]int, len(z.Shape))
	for i := range axes {
		axes[i] = i
	}
	return Sum(z, axes...)
}

func TestTransformerOpParity(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA && !cuda.Available() {
				t.Skip("CUDA unavailable")
			}
			f := loadTransformerFixture(t, device)
			if device == tensor.CUDA {
				EnableGPUProfile(true)
				defer EnableGPUProfile(false)
			}
			run := func(name string, fn func(*testing.T)) {
				t.Run(name, func(t *testing.T) {
					if device == tensor.CUDA {
						ctx, err := NewCUDAContext()
						if err != nil {
							t.Skip(err)
						}
						defer ctx.Close()
					}
					fn(t)
				})
			}
			tol := 2e-4
			if device == tensor.CUDA {
				tol = 2e-3
			}
			for _, name := range []string{"softmax", "log_softmax"} {
				run(name, func(t *testing.T) {
					c := f[name]
					x := fixtureNew(t, c.X, device, true)
					up := fixtureNew(t, c.Up, device, false)
					var y *Tensor
					if name == "softmax" {
						y = Softmax(x, 1)
					} else {
						y = LogSoftmax(x, 1)
					}
					fixtureValues(t, "forward", y, c.Y.Data, tol)
					if e := weightedSum(y, up).Backward(); e != nil {
						t.Fatal(e)
					}
					fixtureGrad(t, "x grad", x, c.X.Grad, tol)
					x.Close()
					up.Close()
				})
			}
			run("dropout", func(t *testing.T) {
				c := f["dropout"]
				x := fixtureNew(t, c.X, device, true)
				up := fixtureNew(t, c.Up, device, false)
				y := Dropout(x, .25, 123, true)
				fixtureValues(t, "forward", y, c.Y.Data, tol)
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "x grad", x, c.X.Grad, tol)
				x.Close()
				up.Close()
			})
			run("cross_entropy", func(t *testing.T) {
				c := f["cross_entropy"]
				x := fixtureNew(t, c.X, device, true)
				y := CrossEntropy(x, c.Target, -100)
				fixtureValues(t, "loss", y, c.Y.Data, tol)
				if e := y.Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "x grad", x, c.X.Grad, tol)
				x.Close()
			})
			run("layer_norm", func(t *testing.T) {
				c := f["layer_norm"]
				x := fixtureNew(t, c.X, device, true)
				w := fixtureNew(t, c.W, device, true)
				b := fixtureNew(t, c.B, device, true)
				up := fixtureNew(t, c.Up, device, false)
				y := LayerNorm(x, w, b, 1e-5)
				fixtureValues(t, "forward", y, c.Y.Data, tol)
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "x grad", x, c.X.Grad, tol)
				fixtureGrad(t, "w grad", w, c.W.Grad, tol)
				fixtureGrad(t, "b grad", b, c.B.Grad, tol)
				x.Close()
				w.Close()
				b.Close()
				up.Close()
			})
			run("batched_matmul", func(t *testing.T) {
				c := f["batched_matmul"]
				a := fixtureNew(t, c.A, device, true)
				b := fixtureNew(t, c.B, device, true)
				up := fixtureNew(t, c.Up, device, false)
				y := MatMul(a, b)
				fixtureValues(t, "forward", y, c.Y.Data, tol)
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "a grad", a, c.A.Grad, tol)
				fixtureGrad(t, "b grad", b, c.B.Grad, tol)
				a.Close()
				b.Close()
				up.Close()
			})
			run("attention", func(t *testing.T) {
				c := f["attention"]
				q := fixtureNew(t, c.Q, device, true)
				k := fixtureNew(t, c.K, device, true)
				v := fixtureNew(t, c.V, device, true)
				mask := fixtureNew(t, c.Mask, device, true)
				up := fixtureNew(t, c.Up, device, false)
				y := ScaledDotProductAttention(q, k, v, mask)
				fixtureValues(t, "forward", y, c.Y.Data, tol)
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "q grad", q, c.Q.Grad, tol)
				fixtureGrad(t, "k grad", k, c.K.Grad, tol)
				fixtureGrad(t, "v grad", v, c.V.Grad, tol)
				fixtureGrad(t, "mask grad", mask, c.Mask.Grad, tol)
				q.Close()
				k.Close()
				v.Close()
				mask.Close()
				up.Close()
			})
			run("embedding_padding", func(t *testing.T) {
				c := f["embedding_padding"]
				w := fixtureNew(t, c.W, device, true)
				up := fixtureNew(t, c.Up, device, false)
				y := EmbeddingWithPadding(w, c.IDs, []int{2, 3}, 0)
				fixtureValues(t, "forward", y, c.Y.Data, tol)
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "w grad", w, c.W.Grad, tol)
				w.Close()
				up.Close()
			})
			run("transformer_stack", func(t *testing.T) {
				c := f["transformer"]
				layers := make([]*TransformerEncoderLayer, 2)
				model := &Module{}
				for i := range layers {
					layer, e := NewTransformerEncoderLayer(8, 2, 16, device)
					if e != nil {
						t.Fatal(e)
					}
					layers[i] = layer
					model.Children = append(model.Children, NamedModule{Name: string(rune('0' + i)), Module: &layer.Module})
				}
				state := map[string][]float32{}
				for name, p := range c.Params {
					state[name] = p.Data
				}
				if e := model.LoadStateDict(state); e != nil {
					t.Fatal(e)
				}
				x := fixtureNew(t, c.X, device, true)
				up := fixtureNew(t, c.Up, device, false)
				y := x
				for _, layer := range layers {
					y = layer.Forward(y, nil, 0)
				}
				fixtureValues(t, "forward", y, c.Y.Data, tol*3)
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				fixtureGrad(t, "input gradient", x, c.X.Grad, tol*4)
				params := model.NamedParameters()
				for _, p := range params {
					fixtureGrad(t, p.Name+" gradient", p.Value, c.Grads[p.Name], tol*4)
				}
				opt := NewAdamW(params, .001, .01)
				opt.Step()
				stepTol := tol * 4
				if device == tensor.CPU {
					stepTol = 2e-3
				} // AdamW amplifies roundoff in near-zero attention bias gradients.
				for _, p := range params {
					fixtureValues(t, p.Name+" AdamW", p.Value, c.Step[p.Name], stepTol)
					p.Value.Close()
				}
				x.Close()
				up.Close()
			})
			if device == tensor.CUDA {
				p := GPUProfile()
				for _, name := range []string{"softmax_f", "softmax_b", "dropout_f", "dropout_b", "ce_f", "ce_b", "embedding_f", "embedding_b", "attn_qk", "attn_softmax_f", "attn_softmax_b", "attn_pv", "layernorm_f", "layernorm_b", "bias_gelu_f", "bias_gelu_b", "matmul_strided"} {
					if p[name].Count == 0 {
						t.Errorf("CUDA kernel %s was not launched", name)
					}
				}
			}
		})
	}
}
