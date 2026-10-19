package autograd

import (
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// TestAuditBF16ShadowLease checks that a graph node holding a raw pointer to a
// cached BF16 shadow also holds a lease on that shadow. Without the lease an
// invalidation between the forward and the backward frees the allocation the
// backward still reads.
func TestAuditBF16ShadowLease(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	BF16Autocast = true
	defer func() { BF16Autocast = false }()

	t.Run("matmul", func(t *testing.T) {
		a, err := New(make([]float32, 2*2*4), []int{2, 2, 4}, tensor.CUDA, true)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()
		b, err := New(make([]float32, 2*4*3), []int{2, 4, 3}, tensor.CUDA, true)
		if err != nil {
			t.Fatal(err)
		}
		defer b.Close()
		y := MatMul(a, b)
		defer y.ReleaseGraph()
		for name, x := range map[string]*Tensor{"input": a, "weight": b} {
			if x.bf16Storage == nil {
				t.Fatalf("%s has no cached BF16 shadow", name)
			}
			if got := x.bf16Storage.references.Load(); got < 2 {
				t.Fatalf("%s BF16 shadow has %d references, want a lease from the graph", name, got)
			}
		}
	})
	t.Run("conv1d", func(t *testing.T) {
		x, err := New(make([]float32, 1*4*2), []int{1, 4, 2}, tensor.CUDA, true)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		w, err := New(make([]float32, 3*2*3), []int{3, 2, 3}, tensor.CUDA, true)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		bias, err := New(make([]float32, 3), []int{3}, tensor.CUDA, true)
		if err != nil {
			t.Fatal(err)
		}
		defer bias.Close()
		y := Conv1d(x, w, bias, 1)
		defer y.ReleaseGraph()
		if w.bf16Storage == nil {
			t.Fatal("weight has no cached BF16 shadow")
		}
		if got := w.bf16Storage.references.Load(); got < 2 {
			t.Fatalf("weight BF16 shadow has %d references, want a lease from the graph", got)
		}
	})
}
