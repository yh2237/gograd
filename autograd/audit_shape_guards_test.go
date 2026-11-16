package autograd

import (
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// A negative dimension can cancel another negative dimension in numel and
// slip past the element-count check.
func TestAuditNegativeShapeRejected(t *testing.T) {
	if _, err := New([]float32{1, 2, 3, 4}, []int{-1, -4}, tensor.CPU, false); err == nil || !strings.Contains(err.Error(), "negative dimension") {
		t.Fatalf("New accepted a negative dimension: %v", err)
	}
	if _, err := Zeros([]int{-1}, tensor.CPU, false); err == nil || !strings.Contains(err.Error(), "negative dimension") {
		t.Fatalf("Zeros accepted a negative dimension: %v", err)
	}
	x := Must([]float32{1, 2, 3, 4}, []int{4}, false)
	defer x.Close()
	for _, shape := range [][]int{{-1, -4}, {-4, -1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("Reshape(%v) was accepted", shape)
				}
			}()
			Reshape(x, shape...)
		}()
	}
	one := Must([]float32{1}, []int{1}, false)
	defer one.Close()
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Expand(-1) was accepted")
			}
		}()
		Expand(one, -1)
	}()
}

// A zero-width channel axis would divide by zero.
func TestAuditMaskedLossRejectsEmptyChannelAxis(t *testing.T) {
	pred := Must(nil, []int{2, 3, 0}, true)
	defer pred.Close()
	target := Must(nil, []int{2, 3, 0}, false)
	defer target.Close()
	defer func() {
		if recover() == nil {
			t.Fatal("MaskedLoss accepted an empty channel axis")
		}
	}()
	MaskedLoss(pred, target, false)
}

// The row walk indexes the target with the prediction's channel count, so a
// target with the same element count but another shape must not be accepted.
func TestAuditMaskedLossRejectsMismatchedTargetShape(t *testing.T) {
	pred := Must(make([]float32, 24), []int{2, 3, 4}, true)
	defer pred.Close()
	target := Must(make([]float32, 24), []int{4, 3, 2}, false)
	defer target.Close()
	defer func() {
		if recover() == nil {
			t.Fatal("MaskedLoss accepted a target with a different shape")
		}
	}()
	MaskedLoss(pred, target, false)
}

// Dimensions that would panic on the first forward are rejected at construction.
func TestAuditLayerConstructorsRejectInvalidDimensions(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, tc := range []struct{ count, width, padding int }{
		{0, 2, -1},
		{2, 0, -1},
		{2, 2, 2},
		{2, 2, -2},
		{-1, 2, -1},
	} {
		if _, err := NewEmbeddingLayer(tc.count, tc.width, tc.padding, tensor.CPU, rng); err == nil {
			t.Fatalf("NewEmbeddingLayer(%d,%d,%d) was accepted", tc.count, tc.width, tc.padding)
		}
	}
	if _, err := NewLayerNormLayer(0, 1e-5, tensor.CPU); err == nil {
		t.Fatal("NewLayerNormLayer(0) was accepted")
	}
	embedding, err := NewEmbeddingLayer(2, 3, 0, tensor.CPU, rng)
	if err != nil {
		t.Fatal(err)
	}
	embedding.Weight.Close()
	norm, err := NewLayerNormLayer(3, 1e-5, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	norm.Weight.Close()
	norm.Bias.Close()
}

// A size-one dim adopts the other length, so broadcasting against a zero dim
// must keep the zero rather than widen it to one.
func TestAuditBroadcastKeepsZeroDimension(t *testing.T) {
	empty := Must([]float32{}, []int{0, 3}, false)
	defer empty.Close()
	one := Must([]float32{1, 2, 3}, []int{1, 3}, false)
	defer one.Close()
	sum := Add(empty, one)
	defer sum.ReleaseGraph()
	if !slices.Equal(sum.Shape, []int{0, 3}) {
		t.Fatalf("Add([0,3],[1,3]) shape = %v, want [0 3]", sum.Shape)
	}

	zeros := Must([]float32{}, []int{0, 0}, false)
	defer zeros.Close()
	ones := Must([]float32{}, []int{1, 0}, false)
	defer ones.Close()
	product := Mul(zeros, ones)
	defer product.ReleaseGraph()
	if !slices.Equal(product.Shape, []int{0, 0}) {
		t.Fatalf("Mul([0,0],[1,0]) shape = %v, want [0 0]", product.Shape)
	}
}

// A batch dim of zero must survive the matmul broadcast instead of becoming one.
func TestAuditMatMulKeepsZeroBatchDimension(t *testing.T) {
	a := Must([]float32{}, []int{0, 0, 3}, false)
	defer a.Close()
	b := Must(make([]float32, 12), []int{1, 3, 4}, false)
	defer b.Close()
	out := MatMul(a, b)
	defer out.ReleaseGraph()
	if !slices.Equal(out.Shape, []int{0, 0, 4}) {
		t.Fatalf("MatMul([0,0,3],[1,3,4]) shape = %v, want [0 0 4]", out.Shape)
	}
}

// A shape whose element count overflows int must not pass the data-length check.
func TestAuditNewRejectsOverflowingShape(t *testing.T) {
	if _, err := New([]float32{}, []int{1 << 32, 1 << 32}, tensor.CPU, true); err == nil {
		t.Fatal("New accepted a shape whose element count overflows int")
	}
	if _, err := Zeros([]int{1 << 32, 1 << 32}, tensor.CPU, false); err == nil {
		t.Fatal("Zeros accepted a shape whose element count overflows int")
	}
}

// The device kernel reads int32 indices, so a wider value must be rejected
// instead of wrapping. The check runs before any allocation.
func TestAuditIndexBufferRejectsWideIndex(t *testing.T) {
	if _, err := NewIndexBuffer([]int{math.MaxInt32 + 1}); err == nil {
		t.Fatal("NewIndexBuffer accepted an index wider than int32")
	}
	if _, err := NewIndexBuffer([]int{-1}); err == nil {
		t.Fatal("NewIndexBuffer accepted a negative index")
	}
}
