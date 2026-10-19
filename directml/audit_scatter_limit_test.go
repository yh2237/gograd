package directml

import (
	"errors"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

// TestAuditDirectMLStridedScatterLimit checks that a very strided scatter is
// rejected instead of building a command list with one region per element.
func TestAuditDirectMLStridedScatterLimit(t *testing.T) {
	b := testBackend(t)
	// The view reads one element per row, so every element needs its own
	// copy region.
	const rows = maxCopyRegions + 1
	base, err := b.Alloc(rows * 2)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	src, err := b.Alloc(rows)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	view := gpu.View{Buffer: base, Shape: []int{rows, 1}, Strides: []int{2, 1}}
	if err := b.Scatter(view, src); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatalf("strided scatter of %d elements: %v, want ErrUnsupported", rows, err)
	}
}
