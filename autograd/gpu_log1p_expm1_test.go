package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func TestGPULog1pExpm1Parity(t *testing.T) {
	for _, device := range []tensor.Device{tensor.Vulkan, tensor.DirectML} {
		t.Run(string(device), func(t *testing.T) {
			if !gpu.Available(string(device)) {
				t.Skipf("%s unavailable", device)
			}
			exec := NewExecutionContext()
			log1pIn := []float32{-0.99, -0.5, -1e-5, -1e-7, 0, 1e-7, 1e-5, 0.5, 10, -1, -2, float32(math.Inf(1)), float32(math.NaN())}
			expm1In := []float32{-30, -5, -0.05, -1e-5, -1e-7, 0, 1e-7, 1e-5, 0.05, 5, 100, float32(math.Inf(-1)), float32(math.Inf(1)), float32(math.NaN())}

			check := func(name string, in []float32, fn func(*Tensor) *Tensor, ref func(float64) float64) {
				t.Helper()
				x, err := exec.New(in, []int{len(in)}, device, false)
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				var y *Tensor
				exec.NoGrad(func() { y = fn(x) })
				defer y.ReleaseGraph()
				got, err := y.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range in {
					want := float32(ref(float64(v)))
					if math.IsNaN(float64(want)) {
						if !math.IsNaN(float64(got[i])) {
							t.Fatalf("%s(%g)=%g want NaN", name, v, got[i])
						}
						continue
					}
					if math.IsInf(float64(want), 0) {
						if !math.IsInf(float64(got[i]), 0) || math.Signbit(float64(got[i])) != math.Signbit(float64(want)) {
							t.Fatalf("%s(%g)=%g want %g", name, v, got[i], want)
						}
						continue
					}
					tol := math.Max(1e-6, math.Abs(float64(want))*1e-5)
					if diff := math.Abs(float64(got[i] - want)); diff > tol {
						t.Fatalf("%s(%g)=%.9g want %.9g (diff %.9g)", name, v, got[i], want, diff)
					}
				}
			}

			check("Log1p", log1pIn, Log1p, math.Log1p)
			check("Expm1", expm1In, Expm1, math.Expm1)
		})
	}
}
