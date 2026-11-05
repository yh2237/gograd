package tensor

import (
	"strings"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

// A negative dimension can cancel another negative dimension in Numel.
func TestAuditReshapeRejectsNegativeDimensions(t *testing.T) {
	tr, err := FromHostOn(CPU, []int{4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Reshape(-1, -4); err == nil || !strings.Contains(err.Error(), "negative dimension") {
		t.Fatalf("Reshape accepted negative dimensions: %v", err)
	}
}

// A Reshape view borrows the parent allocation: closing the view must not free
// the parent's buffer.
func TestAuditReshapeViewCloseKeepsParent(t *testing.T) {
	tr, err := FromHostOn(CPU, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	view, err := tr.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	view.Close()
	got, err := tr.ToHost()
	if err != nil {
		t.Fatalf("parent after view.Close: %v", err)
	}
	for i, want := range []float32{1, 2, 3, 4, 5, 6} {
		if got[i] != want {
			t.Fatalf("parent[%d]=%v want %v", i, got[i], want)
		}
	}
}

// The device path has the same lifetime rule: closing the view used to return
// the shared buffer to the pool and make the parent unusable.
func TestAuditReshapeViewCloseKeepsParentCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	tr, err := FromHost([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	view, err := tr.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	view.Close()
	got, err := tr.ToHost()
	if err != nil {
		t.Fatalf("parent after view.Close: %v", err)
	}
	for i, want := range []float32{1, 2, 3, 4, 5, 6} {
		if got[i] != want {
			t.Fatalf("parent[%d]=%v want %v", i, got[i], want)
		}
	}
}

// A closed handle reports an error instead of dereferencing a nil buffer.
func TestAuditClosedTensorRejectsUse(t *testing.T) {
	tr, err := FromHostOn(CPU, []int{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	tr.Close()
	if _, err := tr.ToHost(); err == nil {
		t.Fatal("ToHost after Close was accepted")
	}
	if err := tr.CopyFrom([]float32{3, 4}); err == nil {
		t.Fatal("CopyFrom after Close was accepted")
	}
	if err := tr.Zero(); err == nil {
		t.Fatal("Zero after Close was accepted")
	}
	if _, err := tr.Reshape(2); err == nil {
		t.Fatal("Reshape after Close was accepted")
	}
	if tr.Data() != nil {
		t.Fatal("Data after Close was not nil")
	}
	if tr.Buffer() != nil {
		t.Fatal("Buffer after Close was not nil")
	}
	tr.Close() // repeated Close stays safe
}
