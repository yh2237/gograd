package autograd

import (
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// MatMul reads a transposed view of a contiguous weight through SGEMM's transB
// path instead of materializing the transpose. The values and both gradients
// must match the hand-computed product.
func TestAuditMatMulTransposedViewGradients(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		a := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, true)
		defer a.Close()
		w := conv2dTestTensor(t, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, []int{4, 3}, device, true)
		defer w.Close()
		y := MatMul(a, Transpose(w, 0, 1))
		if got, want := conv2dHost(t, y, false), []float32{1, 2, 3, 6, 4, 5, 6, 15}; !slices.Equal(got, want) {
			t.Fatalf("output %v, want %v", got, want)
		}
		loss := Sum(y, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		da := conv2dHost(t, a, true)
		dw := conv2dHost(t, w, true)
		loss.ReleaseGraph()
		for i, v := range da {
			if v != 2 {
				t.Fatalf("input gradient[%d] = %g, want 2", i, v)
			}
		}
		if want := []float32{5, 7, 9, 5, 7, 9, 5, 7, 9, 5, 7, 9}; !slices.Equal(dw, want) {
			t.Fatalf("weight gradient %v, want %v", dw, want)
		}
	})
}
