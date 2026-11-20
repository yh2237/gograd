package tensor

import (
	"math"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yh2237/gograd/cuda"
)

func TestOversizedTensorShape(t *testing.T) {
	if _, err := NewOn(CPU, math.MaxInt/4+1); err == nil {
		t.Fatal("oversized CPU allocation accepted")
	}
	tr, err := NewOn(CPU, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Reshape(4, math.MaxInt/2+2); err == nil {
		t.Fatal("reshape accepted an overflowing element count")
	}
}

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

func TestReshapeKeepsStorageAfterParentClose(t *testing.T) {
	for _, device := range []Device{CPU, CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == CUDA && !cuda.Available() {
				t.Skip("cuda unavailable")
			}
			before := cuda.MemoryStats().LiveBytes
			tr, err := FromHostOn(device, []int{2, 2}, []float32{1, 2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}
			defer tr.Close()
			view, err := tr.Reshape(4)
			if err != nil {
				t.Fatal(err)
			}
			defer view.Close()
			tr.Close()
			child, err := view.Reshape(1, 4)
			if err != nil {
				t.Fatal(err)
			}
			defer child.Close()
			view.Close()
			got, err := child.ToHost()
			if err != nil || !slices.Equal(got, []float32{1, 2, 3, 4}) {
				t.Fatalf("view after parent close: %v, %v", got, err)
			}
			child.Close()
			if device == CUDA && cuda.MemoryStats().LiveBytes != before {
				t.Fatal("closing all handles did not release device memory")
			}
		})
	}
}

func TestCUDAReshapeReleasesUnclosedView(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	before := cuda.MemoryStats().LiveBytes
	func() {
		tr, err := NewOn(CUDA, 4)
		if err != nil {
			t.Fatal(err)
		}
		defer tr.Close()
		view, err := tr.Reshape(2, 2)
		if err != nil {
			t.Fatal(err)
		}
		runtime.KeepAlive(view)
	}()
	deadline := time.Now().Add(5 * time.Second)
	for cuda.MemoryStats().LiveBytes != before {
		runtime.GC()
		if time.Now().After(deadline) {
			t.Fatal("unreachable reshape view retained device memory")
		}
		runtime.Gosched()
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
