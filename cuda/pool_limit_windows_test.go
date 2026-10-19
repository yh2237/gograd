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
