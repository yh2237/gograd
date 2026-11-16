package nn

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// NewMaskedLoss normalizes by the weighted element count and rejects negative
// weights before allocating.
func TestNewMaskedLoss(t *testing.T) {
	closeOp := func(op *MaskedLossOp) {
		op.frame.Close()
		if op.channel != nil {
			op.channel.Close()
		}
		op.scalar.Close()
	}

	// Without channel weights the denominator is sum(frame) * channels.
	op, err := NewMaskedLoss(tensor.CPU, []float32{1, 2}, nil, 3, false)
	if err != nil {
		t.Fatal(err)
	}
	if want := float32(1) / 9; math.Abs(float64(op.invDenom-want)) > 1e-6 {
		t.Fatalf("invDenom %g, want %g", op.invDenom, want)
	}
	if op.channel != nil {
		t.Fatal("channel weights allocated without cw")
	}
	if op.channels != 3 || op.l1 {
		t.Fatalf("channels %d l1 %v, want 3 false", op.channels, op.l1)
	}
	closeOp(op)

	// With channel weights the denominator is sum(frame) * sum(channel).
	op, err = NewMaskedLoss(tensor.CPU, []float32{1, 2}, []float32{.5, .5}, 3, true)
	if err != nil {
		t.Fatal(err)
	}
	if want := float32(1) / 3; math.Abs(float64(op.invDenom-want)) > 1e-6 {
		t.Fatalf("invDenom %g, want %g", op.invDenom, want)
	}
	if op.channel == nil {
		t.Fatal("channel weights missing")
	}
	if !op.l1 {
		t.Fatal("l1 flag not stored")
	}
	closeOp(op)

	// An all-zero weight set falls back to a denominator of one.
	op, err = NewMaskedLoss(tensor.CPU, []float32{0}, nil, 4, false)
	if err != nil {
		t.Fatal(err)
	}
	if op.invDenom != 1 {
		t.Fatalf("invDenom %g, want 1", op.invDenom)
	}
	closeOp(op)

	if _, err := NewMaskedLoss(tensor.CPU, []float32{1, -1}, nil, 3, false); err == nil {
		t.Fatal("NewMaskedLoss accepted a negative frame weight")
	}
	if _, err := NewMaskedLoss(tensor.CPU, []float32{1}, []float32{1, -1}, 3, false); err == nil {
		t.Fatal("NewMaskedLoss accepted a negative channel weight")
	}
}
