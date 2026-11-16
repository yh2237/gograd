package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// SmoothL1Loss must match the piecewise definition and its gradient, including
// the unit transition where both branches meet.
func TestAuditSmoothL1Loss(t *testing.T) {
	sinTestDevices(t, func(t *testing.T, device tensor.Device) {
		predicted := []float32{-2, -1, -.5, 0, .5, 1, 2}
		targets := make([]float32, len(predicted))
		x := conv2dTestTensor(t, predicted, []int{len(predicted)}, device, true)
		defer x.Close()
		y := conv2dTestTensor(t, targets, []int{len(targets)}, device, false)
		defer y.Close()
		loss := SmoothL1Loss(x, y, LossOptions{})
		got := conv2dHost(t, loss, false)
		var want float32
		for _, d := range predicted {
			ad := float32(math.Abs(float64(d)))
			if ad < 1 {
				want += .5 * d * d
			} else {
				want += ad - .5
			}
		}
		want /= float32(len(predicted))
		if diff := got[0] - want; diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("loss = %g, want %g", got[0], want)
		}
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		grad := conv2dHost(t, x, true)
		loss.ReleaseGraph()
		for i, d := range predicted {
			wantGrad := d
			if d < -1 {
				wantGrad = -1
			} else if d > 1 {
				wantGrad = 1
			}
			wantGrad /= float32(len(predicted))
			if diff := grad[i] - wantGrad; diff < -1e-6 || diff > 1e-6 {
				t.Fatalf("gradient[%d] = %g, want %g", i, grad[i], wantGrad)
			}
		}
	})
}
