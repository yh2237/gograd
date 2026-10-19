//go:build windows && amd64

package directml

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func submissionSerial(b *Backend) uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.driver.(*nativeDriver).serial
}

func TestDirectMLBatchSubmissions(t *testing.T) {
	b := testBackend(t)
	values := []float32{-3, -1, -.25, 0, .25, 1, 2, 3}
	x := upload(t, b, values)
	y := alloc(t, b, len(values))
	w := upload(t, b, []float32{1, 2, 3, 4})
	bias := upload(t, b, []float32{.1, .2, .3, .4})
	for _, tc := range []struct {
		name  string
		steps uint64
		run   func() error
	}{
		{"gelu", 9, func() error { return b.GELU(y, x, len(values)) }},
		{"layernorm", 3, func() error { return b.LayerNorm(y, x, w, bias, 2, 4, 1e-5) }},
		{"groupnorm", 3, func() error { return b.GroupNorm(y, x, w, bias, 1, 2, 4, 2, 1e-5) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b.ReleaseOperators()
			before := submissionSerial(b)
			if err := tc.run(); err != nil {
				t.Fatal(err)
			}
			if delta := submissionSerial(b) - before; delta != tc.steps+1 {
				t.Fatalf("cold submits=%d, want %d initializers + one execution", delta, tc.steps)
			}
			stats := b.OperatorStats()
			memory := b.MemoryStats()
			for i := 0; i < 4; i++ {
				before = submissionSerial(b)
				if err := tc.run(); err != nil {
					t.Fatal(err)
				}
				if delta := submissionSerial(b) - before; delta != 1 {
					t.Fatalf("warm submits=%d, want one", delta)
				}
			}
			if after := b.OperatorStats(); after != stats {
				t.Fatal("warm batch grew operator cache", stats, after)
			}
			if after := b.MemoryStats(); after != memory {
				t.Fatal("warm batch grew buffer pool", memory, after)
			}
		})
	}
}

func TestDirectMLBatchCachePressure(t *testing.T) {
	b := testBackend(t)
	values := []float32{-3, -1, -.25, 0, .25, 1, 2, 3}
	x := upload(t, b, values)
	y := alloc(t, b, len(values))
	// Make the first GELU step the oldest entry. It must remain pinned when
	// preparation of the second step needs to evict from the full cache.
	if err := b.Mul(y, x, x, len(values)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < maxCachedOperators; i++ {
		if err := b.MulScalar(y, x, len(values), float32(i+10)); err != nil {
			t.Fatal(err)
		}
	}
	if stats := b.OperatorStats(); stats.CachedOperators != maxCachedOperators {
		t.Fatal(stats)
	}
	want := make([]float32, len(values))
	for i, x := range values {
		v := float64(x)
		want[i] = float32(.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+.044715*v*v*v))))
	}
	for i := 0; i < 3; i++ {
		if err := b.GELU(y, x, len(values)); err != nil {
			t.Fatal(err)
		}
		check(t, b, y, want)
		if stats := b.OperatorStats(); stats.CachedOperators > maxCachedOperators || stats.PersistentBytes+stats.TemporaryBytes > operatorCacheBytes {
			t.Fatal(stats)
		}
	}
	b.ReleaseOperators()
	if stats := b.OperatorStats(); stats != (OperatorStats{}) {
		t.Fatal("batch slots not released", stats)
	}
	if err := b.GELU(y, x, len(values)); err != nil {
		t.Fatal(err)
	}
	check(t, b, y, want)
}

func TestDirectMLBatchMoreThanCache(t *testing.T) {
	b := testBackend(t)
	x := upload(t, b, []float32{1, 2, 3, 4})
	y := alloc(t, b, 4)
	z := alloc(t, b, 4)
	// Exercise the uncached path when every cache entry is pinned. Temporary
	// operators must stay alive through execution without exceeding cache bounds.
	run := func() error {
		if err := b.lock(); err != nil {
			return err
		}
		defer b.mu.Unlock()
		buffers := make([]*allocation, 3)
		for i, v := range []*gpu.Buffer{x, y, z} {
			var err error
			buffers[i], err = b.native(v, 4)
			if err != nil {
				return err
			}
		}
		steps := make([]dispatch, maxCachedOperators+2)
		in := buffers[0]
		for i := range steps {
			out := buffers[1+i%2]
			steps[i] = dispatch{opKey{kind: opScalar, n: 4, scalar: math.Float32bits(1), bias: math.Float32bits(.125)}, out, []*allocation{in}}
			in = out
		}
		return b.fail(b.driver.executeBatch(steps))
	}
	for i := 0; i < 2; i++ {
		if err := run(); err != nil {
			t.Fatal(err)
		}
		check(t, b, z, []float32{9.25, 10.25, 11.25, 12.25})
		if stats := b.OperatorStats(); stats.CachedOperators != maxCachedOperators || stats.PersistentBytes+stats.TemporaryBytes > operatorCacheBytes {
			t.Fatal(stats)
		}
	}
	b.ReleaseOperators()
	if stats := b.OperatorStats(); stats != (OperatorStats{}) {
		t.Fatal(stats)
	}
}
