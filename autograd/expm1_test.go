package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestExpm1Parity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := make([]float32, 0, 120)
		for i := -100; i <= 80; i++ {
			values = append(values, float32(i)/10)
		}
		values = append(values, 1e-8, 1e-6, 1e-4, 1e-2)
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Expm1(x)
		defer y.ReleaseGraph()
		if device == tensor.CUDA && (y.Buffer() == nil || y.Data != nil) {
			t.Fatal("Expm1 staged CUDA output on host")
		}
		got := conv2dHost(t, y, false)
		var maxError float64
		for i, v := range values {
			want := math.Expm1(float64(v))
			diff := math.Abs(float64(got[i]) - want)
			maxError = math.Max(maxError, diff)
			if diff > 2e-6*math.Max(1, math.Abs(want)) {
				t.Fatalf("Expm1(%g) = %.9g, want %.9g (error %.9g)", v, got[i], want, diff)
			}
		}
		t.Logf("%s: %d values, max absolute error %.9g", device, len(values), maxError)
	})
}

func TestExpm1KeepsSmallInputs(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1e-8, 1e-7, 1e-6, 1e-5}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Expm1(x)
		defer y.ReleaseGraph()
		got := conv2dHost(t, y, false)
		for i, v := range values {
			if got[i] == 0 {
				t.Fatalf("Expm1(%g) = 0, want a nonzero result", v)
			}
			if math.Abs(float64(got[i]-v)) > 1e-7 {
				t.Fatalf("Expm1(%g) = %g, want approximately %g", v, got[i], v)
			}
		}
	})
}

func TestExpm1NumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-5, -2, -1, -.5, 0, .5, 1, 2, 5}
		weights := []float32{.3, -.7, .2, 1.1, -.4, .9, -.8, .5, 1.3}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
		loss := Sum(Mul(Expm1(x), w), 0)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		evaluate := func(input []float32) float64 {
			probe := conv2dTestTensor(t, input, []int{len(input)}, device, false)
			out := Expm1(probe)
			got := conv2dHost(t, out, false)
			out.ReleaseGraph()
			var sum float64
			for j, v := range got {
				sum += float64(v) * float64(weights[j])
			}
			return sum
		}
		const eps float32 = .001
		var maxError float64
		for i := range values {
			plus := append([]float32(nil), values...)
			minus := append([]float32(nil), values...)
			plus[i] += eps
			minus[i] -= eps
			want := (evaluate(plus) - evaluate(minus)) / float64(plus[i]-minus[i])
			diff := math.Abs(float64(grad[i]) - want)
			maxError = math.Max(maxError, diff)
			if diff > 2e-3*math.Max(1, math.Abs(want)) {
				t.Fatalf("gradient[%d] = %.9g, finite difference %.9g (error %.9g)", i, grad[i], want, diff)
			}
		}
		t.Logf("%s: %d gradients, max absolute error %.9g", device, len(values), maxError)
	})
}

func TestExpm1NonContiguous(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-2, -1, 0, 1, 2, 3}
		x := conv2dTestTensor(t, values, []int{2, 3}, device, true)
		view := Transpose(x, 0, 1)
		if view.IsContiguous() {
			t.Fatal("expected non-contiguous input")
		}
		y := Expm1(view)
		got := conv2dHost(t, y, false)
		order := []int{0, 3, 1, 4, 2, 5}
		for i, j := range order {
			want := math.Expm1(float64(values[j]))
			if math.Abs(float64(got[i])-want) > 2e-6 {
				t.Fatalf("output[%d] = %g, want %g", i, got[i], want)
			}
		}
		loss := Sum(y, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		for i, v := range values {
			want := math.Exp(float64(v))
			if math.Abs(float64(grad[i])-want) > 2e-6 {
				t.Fatalf("gradient[%d] = %g, want %g", i, grad[i], want)
			}
		}
	})
}
