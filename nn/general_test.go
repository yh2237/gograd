package nn

import (
	"bytes"
	"encoding/json"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

func closeEnough(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d != %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] got %.8g want %.8g", label, i, got[i], want[i])
		}
	}
}
func tensorOn(t *testing.T, d tensor.Device, shape []int, v []float32) *tensor.Tensor {
	t.Helper()
	x, e := tensor.FromHostOn(d, shape, v)
	if e != nil {
		t.Fatal(e)
	}
	return x
}
func host(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, e := x.ToHost()
	if e != nil {
		t.Fatal(e)
	}
	return v
}

func TestFiniteDifferences(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	shape := []int{1, 4, 2}
	xv := []float32{-.7, .3, .4, -.2, .8, .1, -.3, .6}
	up := []float32{.2, -.3, .5, .1, -.2, .4, .3, -.1}
	factories := map[string]func() Module{
		"linear": func() Module { m, _ := NewLinearOn(tensor.CPU, 2, 2, rng); return m },
		"conv":   func() Module { m, _ := NewConv1dOn(tensor.CPU, 2, 2, 3, 2, rng); return m },
		"relu":   func() Module { return &ReLU{} }, "leaky": func() Module { return &LeakyReLU{Slope: .2} }, "gelu": func() Module { return &GELU{} }, "tanh": func() Module { return &Tanh{} },
		"residual": func() Module { m, _ := NewConv1dOn(tensor.CPU, 2, 2, 3, 1, rng); return &Residual{Inner: m} },
	}
	for name, makeModule := range factories {
		t.Run(name, func(t *testing.T) {
			m := makeModule()
			if c, ok := m.(interface{ Close() }); ok {
				defer c.Close()
			}
			x := tensorOn(t, tensor.CPU, shape, xv)
			defer x.Close()
			g := tensorOn(t, tensor.CPU, shape, up)
			defer g.Close()
			m.ZeroGrad()
			_, err := m.Forward(nil, x)
			if err != nil {
				t.Fatal(err)
			}
			dx, err := m.Backward(nil, g)
			if err != nil {
				t.Fatal(err)
			}
			analytic := host(t, dx)
			dx.Close()
			objective := func() float64 {
				y, e := m.Forward(nil, x)
				if e != nil {
					t.Fatal(e)
				}
				v := host(t, y)
				var sum float64
				for i := range v {
					sum += float64(v[i] * up[i])
				}
				y.Close()
				return sum
			}
			const eps = .001
			for i := range xv {
				old := x.Data()[i]
				x.Data()[i] = old + eps
				plus := objective()
				x.Data()[i] = old - eps
				minus := objective()
				x.Data()[i] = old
				num := (plus - minus) / (2 * eps)
				if math.Abs(num-float64(analytic[i])) > 2e-3 {
					t.Fatalf("input gradient %d: %g vs %g", i, analytic[i], num)
				}
			}
			if len(m.Parameters()) > 0 {
				p := m.Parameters()[0]
				original := p.Data()[0]
				p.Data()[0] = original + eps
				plus := objective()
				p.Data()[0] = original - eps
				minus := objective()
				p.Data()[0] = original
				num := (plus - minus) / (2 * eps)
				if math.Abs(num-float64(m.Gradients()[0].Data()[0])) > 2e-3 {
					t.Fatalf("weight gradient: %g vs %g", m.Gradients()[0].Data()[0], num)
				}
			}
		})
	}
}

func TestMaskedLossGradients(t *testing.T) {
	p := tensorOn(t, tensor.CPU, []int{1, 2, 2}, []float32{.4, -.3, .8, .2})
	q := tensorOn(t, tensor.CPU, []int{1, 2, 2}, []float32{0, .1, .2, -.1})
	defer p.Close()
	defer q.Close()
	for name, fn := range map[string]func(*tensor.Tensor, *tensor.Tensor, []float32, []float32) (float64, *tensor.Tensor, error){"mse": MaskedWeightedMSE, "l1": MaskedL1} {
		t.Run(name, func(t *testing.T) {
			mask := []float32{1, 0}
			cw := []float32{1, .5}
			_, g, e := fn(p, q, mask, cw)
			if e != nil {
				t.Fatal(e)
			}
			analytic := host(t, g)
			g.Close()
			for i := range p.Data() {
				old := p.Data()[i]
				p.Data()[i] = old + .001
				a, ga, _ := fn(p, q, mask, cw)
				ga.Close()
				p.Data()[i] = old - .001
				b, gb, _ := fn(p, q, mask, cw)
				gb.Close()
				p.Data()[i] = old
				if math.Abs((a-b)/.002-float64(analytic[i])) > 1e-4 {
					t.Fatalf("gradient %d", i)
				}
			}
		})
	}
}

type fixture struct {
	Input, Target, Mask, Channels, Forward []float32
	Loss                                   float64
	Parameters, Gradients, Updated         map[string][]float32
}

func TestTorchConvParity(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA && !cuda.Available() {
				t.Skip("CUDA unavailable")
			}
			if device == tensor.CUDA {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				if e := cuda.SetDevice(0); e != nil {
					t.Skip(e)
				}
			}
			var blas *cuda.Blas
			if device == tensor.CUDA {
				var e error
				blas, e = cuda.NewBlas()
				if e != nil {
					t.Fatal(e)
				}
				defer blas.Destroy()
			}
			raw, e := os.ReadFile("../testdata/nn_conv_step.json")
			if e != nil {
				t.Fatal(e)
			}
			var f fixture
			if e = json.Unmarshal(raw, &f); e != nil {
				t.Fatal(e)
			}
			a, _ := NewLinearOn(device, 2, 3, rand.New(rand.NewSource(1)))
			c, _ := NewConv1dOn(device, 3, 3, 3, 2, rand.New(rand.NewSource(1)))
			z, _ := NewLinearOn(device, 3, 2, rand.New(rand.NewSource(1)))
			model := &Sequential{Modules: []Module{a, &Residual{Inner: &Sequential{Modules: []Module{c, &GELU{}}}}, z}}
			defer model.Close()
			names := []string{"input.weight", "input.bias", "conv.weight", "conv.bias", "output.weight", "output.bias"}
			for i, p := range model.Parameters() {
				if e := p.CopyFrom(f.Parameters[names[i]]); e != nil {
					t.Fatal(e)
				}
			}
			x := tensorOn(t, device, []int{2, 5, 2}, f.Input)
			defer x.Close()
			target := tensorOn(t, device, []int{2, 5, 2}, f.Target)
			defer target.Close()
			y, e := model.Forward(blas, x)
			if e != nil {
				t.Fatal(e)
			}
			closeEnough(t, "forward", host(t, y), f.Forward, 3e-5)
			loss, g, e := MaskedWeightedMSE(y, target, f.Mask, f.Channels)
			if e != nil {
				t.Fatal(e)
			}
			if math.Abs(loss-f.Loss) > 1e-5 {
				t.Fatalf("loss %g != %g", loss, f.Loss)
			}
			model.ZeroGrad()
			dx, e := model.Backward(blas, g)
			if e != nil {
				t.Fatal(e)
			}
			dx.Close()
			g.Close()
			model.CloseActivations()
			for i, p := range model.Gradients() {
				closeEnough(t, names[i]+" grad", host(t, p), f.Gradients[names[i]], 5e-5)
			}
			opt, e := NewAdamW(model.Parameters(), model.Gradients(), .003, .01, .8)
			if e != nil {
				t.Fatal(e)
			}
			defer opt.Close()
			if e = opt.Step(); e != nil {
				t.Fatal(e)
			}
			for i, p := range model.Parameters() {
				closeEnough(t, names[i]+" update", host(t, p), f.Updated[names[i]], 5e-5)
			}
			var b bytes.Buffer
			if e = Save(&b, model); e != nil {
				t.Fatal(e)
			}
			if _, e = Load(&b, model); e != nil {
				t.Fatal(e)
			}
		})
	}
}
