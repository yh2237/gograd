package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestClampParity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		nan := float32(math.NaN())
		values := []float32{-10, -1, -.5, 0, .5, 1, 10, float32(math.Inf(-1)), float32(math.Inf(1)), nan}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		for _, bounds := range [][2]float32{{-1, 1}, {nan, 1}, {-1, nan}} {
			lo, hi := bounds[0], bounds[1]
			y := Clamp(x, lo, hi)
			got := conv2dHost(t, y, false)
			y.ReleaseGraph()
			for i, v := range values {
				want := min(max(v, lo), hi)
				if got[i] != want && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(want))) {
					t.Errorf("Clamp(%g, %g, %g) = %g, want %g", v, lo, hi, got[i], want)
				}
			}
		}
	})
}

func TestClampNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-3, -.5, 0, .5, 3}
		weights := []float32{.3, -.7, .2, 1.1, -.4}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
		loss := Sum(Mul(Clamp(x, -1, 1), w), 0)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		for i, v := range values {
			want := float32(0)
			if v >= -1 && v <= 1 {
				want = weights[i]
			}
			if grad[i] != want {
				t.Fatalf("gradient[%d] = %g, want %g", i, grad[i], want)
			}
		}
	})
}

func TestPowScalarParity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{.25, .5, 1, 2, 4, 9}
		for _, p := range []float32{0, .5, 1, 2, 3, -1} {
			x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
			y := PowScalar(x, p)
			got := conv2dHost(t, y, false)
			y.ReleaseGraph()
			for i, v := range values {
				want := math.Pow(float64(v), float64(p))
				if diff := math.Abs(float64(got[i]) - want); diff > 1e-6*math.Max(1, math.Abs(want)) {
					t.Fatalf("PowScalar(%g, %g) = %.9g, want %.9g", v, p, got[i], want)
				}
			}
		}
	})
}

func TestPowScalarNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{.5, 1, 2, 3}
		weights := []float32{.3, -.7, .2, 1.1}
		for _, p := range []float32{.5, 2, 3, -1} {
			x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
			w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
			loss := Sum(Mul(PowScalar(x, p), w), 0)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			grad := conv2dHost(t, x, true)
			loss.ReleaseGraph()
			const eps float32 = .001
			for i, v := range values {
				hi := math.Pow(float64(v+eps), float64(p))
				lo := math.Pow(float64(v-eps), float64(p))
				want := (hi - lo) / float64(2*eps) * float64(weights[i])
				if diff := math.Abs(float64(grad[i]) - want); diff > 1e-3*math.Max(1, math.Abs(want)) {
					t.Fatalf("p=%g gradient[%d] = %.9g, finite difference %.9g", p, i, grad[i], want)
				}
			}
		}
	})
}
