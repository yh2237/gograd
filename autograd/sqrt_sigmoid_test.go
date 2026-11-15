package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestSqrtParity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{0, 1e-8, 1e-4, .25, 1, 2, 9, 1e4, 1e8}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Sqrt(x)
		defer y.ReleaseGraph()
		got := conv2dHost(t, y, false)
		for i, v := range values {
			want := math.Sqrt(float64(v))
			if diff := math.Abs(float64(got[i]) - want); diff > 1e-6*math.Max(1, want) {
				t.Fatalf("Sqrt(%g) = %.9g, want %.9g", v, got[i], want)
			}
		}
	})
}

func TestSqrtNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{.25, 1, 2, 7.5, 100, 1e4}
		weights := []float32{.3, -.7, .2, 1.1, -.4, .9}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
		loss := Sum(Mul(Sqrt(x), w), 0)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		const eps float32 = .001
		for i := range values {
			plus := append([]float32(nil), values...)
			minus := append([]float32(nil), values...)
			plus[i] += eps
			minus[i] -= eps
			want := (math.Sqrt(float64(plus[i])) - math.Sqrt(float64(minus[i]))) / float64(plus[i]-minus[i]) * float64(weights[i])
			if diff := math.Abs(float64(grad[i]) - want); diff > 1e-4 {
				t.Fatalf("gradient[%d] = %.9g, finite difference %.9g", i, grad[i], want)
			}
		}
	})
}

func TestSigmoidParity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-100, -20, -8, -1, -.5, 0, .5, 1, 8, 20, 100}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Sigmoid(x)
		defer y.ReleaseGraph()
		got := conv2dHost(t, y, false)
		for i, v := range values {
			want := 1 / (1 + math.Exp(-float64(v)))
			if diff := math.Abs(float64(got[i]) - want); diff > 1e-6 {
				t.Fatalf("Sigmoid(%g) = %.9g, want %.9g", v, got[i], want)
			}
		}
	})
}

func TestSigmoidNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-6, -2, -.5, 0, .5, 2, 6}
		weights := []float32{.3, -.7, .2, 1.1, -.4, .9, -.8}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
		loss := Sum(Mul(Sigmoid(x), w), 0)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		const eps float32 = .001
		for i := range values {
			plus := append([]float32(nil), values...)
			minus := append([]float32(nil), values...)
			plus[i] += eps
			minus[i] -= eps
			hi := 1 / (1 + math.Exp(-float64(plus[i])))
			lo := 1 / (1 + math.Exp(-float64(minus[i])))
			want := (hi - lo) / float64(plus[i]-minus[i]) * float64(weights[i])
			if diff := math.Abs(float64(grad[i]) - want); diff > 1e-4 {
				t.Fatalf("gradient[%d] = %.9g, finite difference %.9g", i, grad[i], want)
			}
		}
	})
}
