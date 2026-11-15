package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// A nil bias must behave exactly like a zero bias, in the forward pass and in
// the weight and input gradients.
func TestAuditConv1dNilBias(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		values := []float32{1, 2, 3, 4, 5, 6}
		weights := []float32{.1, .2, .3, .4, .5, .6}
		for name, apply := range map[string]func(x, w, b *Tensor) *Tensor{
			"conv1d":             func(x, w, b *Tensor) *Tensor { return Conv1d(x, w, b, 1) },
			"conv1d_gemm":        func(x, w, b *Tensor) *Tensor { return Conv1dGEMM(x, w, b, 1) },
			"conv1d_gemm_stride": func(x, w, b *Tensor) *Tensor { return Conv1dGEMMStride(x, w, b, 1, 1, 1) },
		} {
			t.Run(name, func(t *testing.T) {
				run := func(bias *Tensor) ([]float32, []float32, []float32) {
					x := conv2dTestTensor(t, values, []int{1, 3, 2}, device, true)
					w := conv2dTestTensor(t, weights, []int{1, 2, 3}, device, true)
					y := apply(x, w, bias)
					out := conv2dHost(t, y, false)
					loss := Sum(y, 0, 1, 2)
					if err := loss.Backward(); err != nil {
						t.Fatal(err)
					}
					dx := conv2dHost(t, x, true)
					dw := conv2dHost(t, w, true)
					loss.ReleaseGraph()
					x.Close()
					w.Close()
					return out, dx, dw
				}
				zero := conv2dTestTensor(t, []float32{0}, []int{1}, device, false)
				defer zero.Close()
				nilOut, nilDX, nilDW := run(nil)
				zeroOut, zeroDX, zeroDW := run(zero)
				if !slices.Equal(nilOut, zeroOut) {
					t.Fatalf("nil bias output %v != zero bias %v", nilOut, zeroOut)
				}
				if !slices.Equal(nilDX, zeroDX) {
					t.Fatalf("nil bias input gradient %v != zero bias %v", nilDX, zeroDX)
				}
				if !slices.Equal(nilDW, zeroDW) {
					t.Fatalf("nil bias weight gradient %v != zero bias %v", nilDW, zeroDW)
				}
			})
		}
	})
}
