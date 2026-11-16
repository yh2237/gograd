package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func softplusReference(x float64) float64 {
	return math.Max(x, 0) + math.Log1p(math.Exp(-math.Abs(x)))
}

func TestSoftplusParity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := make([]float32, 0, 161)
		for i := -80; i <= 80; i++ {
			values = append(values, float32(i)/2)
		}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Softplus(x)
		defer y.ReleaseGraph()
		if device == tensor.CUDA && (y.Buffer() == nil || y.Data != nil) {
			t.Fatal("Softplus staged CUDA output on host")
		}
		got := conv2dHost(t, y, false)
		var maxError float64
		for i, v := range values {
			want := softplusReference(float64(v))
			diff := math.Abs(float64(got[i]) - want)
			maxError = math.Max(maxError, diff)
			if diff > 2e-5 {
				t.Fatalf("Softplus(%g) = %.9g, want %.9g (error %.9g)", v, got[i], want, diff)
			}
		}
		t.Logf("%s: %d values, max absolute error %.9g", device, len(values), maxError)
	})
}

func TestSoftplusLargeMagnitude(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-1e30, -1000, -100, 100, 1000, 1e30}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Softplus(x)
		defer y.ReleaseGraph()
		got := conv2dHost(t, y, false)
		for i, v := range values {
			if math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(got[i])) {
				t.Fatalf("Softplus(%g) = %g, want a finite value", v, got[i])
			}
			want := softplusReference(float64(v))
			if math.Abs(float64(got[i])-want) > 1e-5*math.Max(1, math.Abs(want)) {
				t.Fatalf("Softplus(%g) = %.9g, want %.9g", v, got[i], want)
			}
		}
	})
}

func TestSoftplusNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-20, -10.25, -3.1, -1, -.1, 0, .4, 1.7, 6.2, 13, 20}
		weights := []float32{.3, -.7, .2, 1.1, -.4, .9, -.8, .5, 1.3, -.2, .6}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
		loss := Sum(Mul(Softplus(x), w), 0)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		evaluate := func(input []float32) float64 {
			probe := conv2dTestTensor(t, input, []int{len(input)}, device, false)
			out := Softplus(probe)
			got := conv2dHost(t, out, false)
			out.ReleaseGraph()
			var sum float64
			for j, v := range got {
				sum += float64(v) * float64(weights[j])
			}
			return sum
		}
		const eps float32 = .01
		var maxError float64
		for i := range values {
			plus := append([]float32(nil), values...)
			minus := append([]float32(nil), values...)
			plus[i] += eps
			minus[i] -= eps
			want := (evaluate(plus) - evaluate(minus)) / float64(plus[i]-minus[i])
			diff := math.Abs(float64(grad[i]) - want)
			maxError = math.Max(maxError, diff)
			if diff > 2e-4 {
				t.Fatalf("gradient[%d] = %.9g, finite difference %.9g (error %.9g)", i, grad[i], want, diff)
			}
		}
		t.Logf("%s: %d gradients, max absolute error %.9g", device, len(values), maxError)
	})
}
