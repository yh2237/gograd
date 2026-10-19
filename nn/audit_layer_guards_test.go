package nn

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// TestAuditBackwardBeforeForward checks that the layers report a misuse as an
// error instead of dereferencing a nil saved tensor.
func TestAuditBackwardBeforeForward(t *testing.T) {
	grad, err := tensor.ZerosOn(tensor.CPU, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer grad.Close()
	linear, err := NewLinearOn(tensor.CPU, 2, 2, rand.New(rand.NewSource(3)))
	if err != nil {
		t.Fatal(err)
	}
	defer linear.Close()
	if _, err := linear.Backward(nil, grad); err == nil || !strings.Contains(err.Error(), "before forward") {
		t.Fatalf("linear backward before forward: %v", err)
	}
	tanh := &Tanh{}
	if _, err := tanh.Backward(nil, grad); err == nil || !strings.Contains(err.Error(), "before forward") {
		t.Fatalf("tanh backward before forward: %v", err)
	}
}

// TestAuditMSERejectsMismatchedInputs checks that the loss helper reports
// unusable inputs instead of panicking or returning NaN.
func TestAuditMSERejectsMismatchedInputs(t *testing.T) {
	if loss, gradient := MSE([]float32{1, 2}, []float32{1}); loss != 0 || gradient != nil {
		t.Fatalf("mismatched MSE = %v, %v", loss, gradient)
	}
	if loss, gradient := MSE(nil, nil); loss != 0 || gradient != nil {
		t.Fatalf("empty MSE = %v, %v", loss, gradient)
	}
	loss, gradient := MSE([]float32{1, 3}, []float32{1, 1})
	if loss != 2 || len(gradient) != 2 || gradient[0] != 0 || gradient[1] != 2 {
		t.Fatalf("MSE = %v, %v", loss, gradient)
	}
}
