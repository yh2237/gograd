package autograd

import (
	"math/rand"
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
