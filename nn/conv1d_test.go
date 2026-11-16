package nn

import (
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// NewConv1d rejects dimensions the symmetric-padding forward cannot use and
// builds a weight of [out,in,kernel] plus a bias of [out].
func TestNewConv1d(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, bad := range [][4]int{{0, 2, 3, 1}, {2, 0, 3, 1}, {2, 2, 0, 1}, {2, 2, 4, 1}, {2, 2, 3, 0}} {
		if _, err := NewConv1dOn(tensor.CPU, bad[0], bad[1], bad[2], bad[3], rng); err == nil {
			t.Fatalf("NewConv1dOn accepted in=%d out=%d kernel=%d dilation=%d", bad[0], bad[1], bad[2], bad[3])
		}
	}

	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	c, err := NewConv1d(2, 3, 5, 2, rng)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	params := c.Parameters()
	if len(params) != 2 {
		t.Fatalf("parameter count %d, want 2", len(params))
	}
	if shape := params[0].Shape(); len(shape) != 3 || shape[0] != 3 || shape[1] != 2 || shape[2] != 5 {
		t.Fatalf("weight shape %v, want [3 2 5]", shape)
	}
	if shape := params[1].Shape(); len(shape) != 1 || shape[0] != 3 {
		t.Fatalf("bias shape %v, want [3]", shape)
	}
}
