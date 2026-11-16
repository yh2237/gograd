package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestScalarOpsValues(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-4, -1, 0, .5, 3, 10}
		const v float32 = 2
		for _, tc := range []struct {
			name string
			op   func(*Tensor, float32) *Tensor
			want func(float32) float32
		}{
			{"AddScalar", AddScalar, func(x float32) float32 { return x + v }},
			{"SubScalar", SubScalar, func(x float32) float32 { return x - v }},
			{"MulScalar", MulScalar, func(x float32) float32 { return x * v }},
			{"DivScalar", DivScalar, func(x float32) float32 { return x / v }},
		} {
			x := conv2dTestTensor(t, values, []int{len(values)}, device, false)
			y := tc.op(x, v)
			got := conv2dHost(t, y, false)
			for i, in := range values {
				if want := tc.want(in); got[i] != want {
					t.Fatalf("%s(%g) = %g, want %g", tc.name, in, got[i], want)
				}
			}
			y.ReleaseGraph()
		}
	})
}

func TestScalarOpsGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-4, -1, 0, .5, 3, 10}
		const v float32 = 2
		for _, tc := range []struct {
			name string
			op   func(*Tensor, float32) *Tensor
			want float32
		}{
			{"AddScalar", AddScalar, 1},
			{"SubScalar", SubScalar, 1},
			{"MulScalar", MulScalar, v},
			{"DivScalar", DivScalar, 1 / v},
		} {
			x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
			loss := Sum(tc.op(x, v), 0)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			grad := conv2dHost(t, x, true)
			for i := range values {
				if grad[i] != tc.want {
					t.Fatalf("%s gradient[%d] = %g, want %g", tc.name, i, grad[i], tc.want)
				}
			}
			loss.ReleaseGraph()
		}
	})
}

func TestScalarOpsNumericGradient(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{-4, -1, .5, 3, 10}
		weights := []float32{.3, -.7, 1.1, -.4, .9}
		const v float32 = 2.5
		for _, tc := range []struct {
			name string
			op   func(*Tensor, float32) *Tensor
		}{
			{"AddScalar", AddScalar},
			{"SubScalar", SubScalar},
			{"MulScalar", MulScalar},
			{"DivScalar", DivScalar},
		} {
			x := conv2dTestTensor(t, values, []int{len(values)}, device, true)
			w := conv2dTestTensor(t, weights, []int{len(weights)}, device, false)
			loss := Sum(Mul(tc.op(x, v), w), 0)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			grad := conv2dHost(t, x, true)
			evaluate := func(input []float32) float64 {
				probe := conv2dTestTensor(t, input, []int{len(input)}, device, false)
				out := tc.op(probe, v)
				got := conv2dHost(t, out, false)
				out.ReleaseGraph()
				var sum float64
				for j, g := range got {
					sum += float64(g) * float64(weights[j])
				}
				return sum
			}
			const eps float32 = .01
			for i := range values {
				plus := append([]float32(nil), values...)
				minus := append([]float32(nil), values...)
				plus[i] += eps
				minus[i] -= eps
				want := (evaluate(plus) - evaluate(minus)) / float64(plus[i]-minus[i])
				if diff := math.Abs(float64(grad[i]) - want); diff > 2e-4 {
					t.Fatalf("%s gradient[%d] = %.9g, finite difference %.9g (error %.9g)", tc.name, i, grad[i], want, diff)
				}
			}
			loss.ReleaseGraph()
		}
	})
}
