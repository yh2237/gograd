//go:build windows

package cuda

import "testing"

// TestDefaultPoolCacheLimitFitsPlatformInt checks the platform invariant: the
// default cache limit must be representable as an int on every platform. The
// conversion truncates silently on 32-bit builds.
func TestDefaultPoolCacheLimitFitsPlatformInt(t *testing.T) {
	maximum := int64(^uint(0) >> 1)
	got := initialPoolCacheLimit()
	if got <= 0 || int64(got) > maximum {
		t.Fatalf("initial pool cache limit = %d, want a positive value within %d", got, maximum)
	}
	if int64(defaultPoolCacheLimit) <= maximum && int64(got) != defaultPoolCacheLimit {
		t.Fatalf("initial pool cache limit = %d, want %d", got, defaultPoolCacheLimit)
	}
}

func TestEmptyAllocationsDoNotEnterPool(t *testing.T) {
	if !Available() {
		t.Skip("CUDA unavailable")
	}
	if err := ReleasePool(); err != nil {
		t.Fatal(err)
	}
	before := MemoryStats()
	first, err := Alloc(0)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Free()
	second, err := Alloc(0)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Free()
	if err := second.Free(); err != nil {
		t.Fatal(err)
	}
	if err := first.Free(); err != nil {
		t.Fatal(err)
	}
	after := MemoryStats()
	if after.CachedBuffers != before.CachedBuffers || after.LiveBytes != before.LiveBytes {
		t.Fatalf("empty allocations changed memory stats: %+v -> %+v", before, after)
	}
}
