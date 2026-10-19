//go:build windows

package cuda

import "testing"

func TestDefaultPoolCacheLimitFitsPlatformInt(t *testing.T) {
	maximum := int64(^uint(0) >> 1)
	want := defaultPoolCacheLimit
	if want > maximum {
		want = maximum
	}
	if got := initialPoolCacheLimit(); got <= 0 || int64(got) != want {
		t.Fatalf("initial pool cache limit = %d, want %d", got, want)
	}
}
