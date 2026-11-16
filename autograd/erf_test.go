package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestErfParity(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := make([]float32, 0, 161)
		for i := -80; i <= 80; i++ {
			values = append(values, float32(i)/8)
		}
		values = append(values, -1e10, -100, 100, 1e10)
		x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
		y := Erf(x)
		defer y.ReleaseGraph()
		if device == tensor.CUDA && (y.Buffer() == nil || y.Data != nil) {
			t.Fatal("Erf staged CUDA output on host")
		}
		got := conv2dHost(t, y, false)
		var maxError float64
		for i, v := range values {
			want := math.Erf(float64(v))
			diff := math.Abs(float64(got[i]) - want)
			maxError = math.Max(maxError, diff)
			if diff > 2e-6 {
				t.Fatalf("Erf(%g) = %.9g, want %.9g (error %.9g)", v, got[i], want, diff)
			}
		}
		t.Logf("%s: %d values, max absolute error %.9g", device, len(values), maxError)
	})
}

func TestErfNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-6, -3.1, -1.7, -1, -.1, 0, .4, 1.7, 3.1, 6}
		weights := []float32{.3, -.7, .2, 1.1, -.4, .9, -.8, .5, 1.3, -.2}
		x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
		w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
		loss := Sum(Mul(Erf(x), w), 0)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		defer loss.ReleaseGraph()
		grad := conv2dHost(t, x, true)
		evaluate := func(input []float32) float64 {
			probe := conv2dTestTensor(t, input, []int{len(input)}, device, false)
			out := Erf(probe)
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

func TestErfNonContiguous(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-2, -1, 0, 1, 2, 3}
		x := conv2dTestTensor(t, values, []int{2, 3}, device, true)
		view := Transpose(x, 0, 1)
		if view.IsContiguous() {
			t.Fatal("expected non-contiguous input")
		}
		y := Erf(view)
		got := conv2dHost(t, y, false)
		order := []int{0, 3, 1, 4, 2, 5}
		for i, j := range order {
			want := math.Erf(float64(values[j]))
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
			want := 2 / math.Sqrt(math.Pi) * math.Exp(-float64(v)*float64(v))
			if math.Abs(float64(grad[i])-want) > 2e-6 {
				t.Fatalf("gradient[%d] = %g, want %g", i, grad[i], want)
			}
		}
	})
}
