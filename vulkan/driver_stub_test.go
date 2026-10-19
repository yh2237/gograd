//go:build (!windows && !linux) || android || !amd64

package vulkan

import (
	"errors"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func TestVulkanUnsupportedPlatform(t *testing.T) {
	b := NewBackend()
	if b.Available() || !errors.Is(b.Err(), gpu.ErrUnavailable) {
		t.Fatalf("unexpected availability: %v", b.Err())
	}
	if _, err := b.Alloc(1); !errors.Is(err, gpu.ErrUnavailable) {
		t.Fatal(err)
	}
	if s := b.MemoryStats(); s.ReservedBytes != 0 || s.LiveBuffers != 0 {
		t.Fatalf("stub allocated memory: %+v", s)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}
