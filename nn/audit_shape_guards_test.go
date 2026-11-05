package nn

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// A zero or negative feature count would divide by zero in Forward.
func TestAuditLinearRejectsInvalidDimensions(t *testing.T) {
	for _, tc := range []struct{ in, out int }{{0, 2}, {2, 0}, {-1, 2}, {2, -1}} {
		if _, err := NewLinearOn(tensor.CPU, tc.in, tc.out, rand.New(rand.NewSource(1))); err == nil {
			t.Fatalf("NewLinearOn(%d,%d) was accepted", tc.in, tc.out)
		}
	}
}

// A gradient with the wrong shape or device would panic inside SGEMM.
func TestAuditLinearBackwardRejectsMismatchedGradient(t *testing.T) {
	linear, err := NewLinearOn(tensor.CPU, 3, 4, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer linear.Close()
	x, err := tensor.FromHostOn(tensor.CPU, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err := linear.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	for _, shape := range [][]int{{1, 2}, {2, 3}, {2, 4, 1}} {
		g, err := tensor.ZerosOn(tensor.CPU, shape...)
		if err != nil {
			t.Fatal(err)
		}
		_, err = linear.Backward(nil, g)
		g.Close()
		if err == nil || !strings.Contains(err.Error(), "gradient shape") {
			t.Fatalf("gradient %v: %v", shape, err)
		}
	}
	if _, err := linear.Backward(nil, nil); err == nil {
		t.Fatal("nil gradient was accepted")
	}
}

// Tanh uses the same guard.
func TestAuditTanhBackwardRejectsMismatchedGradient(t *testing.T) {
	tanh := &Tanh{}
	x, err := tensor.FromHostOn(tensor.CPU, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err := tanh.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	g, err := tensor.ZerosOn(tensor.CPU, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, err := tanh.Backward(nil, g); err == nil || !strings.Contains(err.Error(), "gradient shape") {
		t.Fatalf("Tanh accepted a mismatched gradient: %v", err)
	}
	if _, err := tanh.Backward(nil, nil); err == nil {
		t.Fatal("Tanh accepted a nil gradient")
	}
}

// wrongGradientModule returns a gradient of a different shape than it received,
// which is what the Residual container has to reject.
type wrongGradientModule struct{ shape []int }

func (m *wrongGradientModule) Parameters() []*tensor.Tensor { return nil }
func (m *wrongGradientModule) Gradients() []*tensor.Tensor  { return nil }
func (m *wrongGradientModule) ZeroGrad()                    {}
func (m *wrongGradientModule) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	return x, nil
}
func (m *wrongGradientModule) Backward(_ *cuda.Blas, _ *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.ZerosOn(tensor.CPU, m.shape...)
}

// A container must not add a gradient of the wrong shape to its skip connection.
func TestAuditResidualBackwardRejectsMismatchedGradient(t *testing.T) {
	residual := &Residual{Inner: &wrongGradientModule{shape: []int{1}}}
	x, err := tensor.FromHostOn(tensor.CPU, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err := residual.Forward(nil, x); err != nil {
		t.Fatal(err)
	}
	g, err := tensor.ZerosOn(tensor.CPU, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, err := residual.Backward(nil, g); err == nil || !strings.Contains(err.Error(), "gradient shape") {
		t.Fatalf("Residual accepted a mismatched gradient: %v", err)
	}
	if _, err := residual.Backward(nil, nil); err == nil {
		t.Fatal("Residual accepted a nil gradient")
	}
}

// A nil adjoint is reported before it reaches a module.
func TestAuditSequentialBackwardRejectsNilGradient(t *testing.T) {
	seq := &Sequential{Modules: []Module{&Tanh{}}}
	if _, err := seq.Backward(nil, nil); err == nil {
		t.Fatal("Sequential accepted a nil gradient")
	}
}

// Optimizer state must not be built for a gradient whose shape or device
// differs from its parameter.
func TestAuditAdamWRejectsMismatchedGradient(t *testing.T) {
	param, err := tensor.FromHostOn(tensor.CPU, []int{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer param.Close()
	grad, err := tensor.FromHostOn(tensor.CPU, []int{1}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer grad.Close()
	if _, err := NewAdamW([]*tensor.Tensor{param}, []*tensor.Tensor{grad}, .1, 0, 0); err == nil {
		t.Fatal("AdamW accepted a mismatched gradient")
	}
	if _, err := NewAdamW([]*tensor.Tensor{param}, []*tensor.Tensor{nil}, .1, 0, 0); err == nil {
		t.Fatal("AdamW accepted a nil gradient")
	}
}
