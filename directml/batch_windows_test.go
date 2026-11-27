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
		name      string
		uniqueOps uint64
		run       func() error
	}{
		{"gelu", 0, func() error { return b.GELU(y, x, len(values)) }},
		{"layernorm", 1, func() error { return b.LayerNorm(y, x, w, bias, 2, 4, 1e-5) }},
		{"groupnorm", 1, func() error { return b.GroupNorm(y, x, w, bias, 1, 2, 4, 2, 1e-5) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b.ReleaseOperators()
			before := submissionSerial(b)
			if err := tc.run(); err != nil {
				t.Fatal(err)
			}
			if submissionSerial(b) != before {
				t.Fatalf("op flushed before Synchronize")
			}
			if err := b.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if delta := submissionSerial(b) - before; delta != tc.uniqueOps+1 {
				t.Fatalf("cold submits=%d, want %d initializers + one execution", delta, tc.uniqueOps)
			}
			stats := b.OperatorStats()
			memory := b.MemoryStats()
			for i := 0; i < 4; i++ {
				before = submissionSerial(b)
				if err := tc.run(); err != nil {
					t.Fatal(err)
				}
				if err := b.Synchronize(); err != nil {
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
			before = submissionSerial(b)
			for i := 0; i < 4; i++ {
				if err := tc.run(); err != nil {
					t.Fatal(err)
				}
			}
			if err := b.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if delta := submissionSerial(b) - before; delta != 1 {
				t.Fatalf("multi-op deferred submits=%d, want 1", delta)
			}
		})
	}
}

func TestDirectMLBatchCachePressure(t *testing.T) {
	b := testBackend(t)
	values := []float32{-3, -1, -.25, 0, .25, 1, 2, 3}
	x := upload(t, b, values)
	y := alloc(t, b, len(values))
	if err := b.Mul(y, x, x, len(values)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < maxCachedOperators; i++ {
		if err := b.MulScalar(y, x, len(values), float32(i+10)); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Synchronize(); err != nil {
		t.Fatal(err)
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
			steps[i] = dispatch{opKey{kind: opScalar, n: 4, scalar: math.Float32bits(1), bias: math.Float32bits(float32(i+1) * .125)}, out, []*allocation{in}}
			in = out
		}
		if err := b.fail(b.driver.executeBatch(steps)); err != nil {
			return err
		}
		return b.flushLocked()
	}
	sumBias := float32((maxCachedOperators+2)*(maxCachedOperators+3)/2) * .125
	want := []float32{1 + sumBias, 2 + sumBias, 3 + sumBias, 4 + sumBias}
	for i := 0; i < 2; i++ {
		if err := run(); err != nil {
			t.Fatal(err)
		}
		check(t, b, z, want)
		if stats := b.OperatorStats(); stats.CachedOperators != maxCachedOperators || stats.PersistentBytes+stats.TemporaryBytes > operatorCacheBytes {
			t.Fatal(stats)
		}
	}
	b.ReleaseOperators()
	if stats := b.OperatorStats(); stats != (OperatorStats{}) {
		t.Fatal(stats)
	}
}

func TestDirectMLDeferredBatchAndInterleavedCopy(t *testing.T) {
	b := testBackend(t)
	x := upload(t, b, []float32{1, 2, 3, 4})
	y := alloc(t, b, 4)
	z := alloc(t, b, 4)
	if err := b.Add(y, x, x, 4); err != nil {
		t.Fatal(err)
	}
	if err := b.Clamp(z, y, 4, -1, 6); err != nil {
		t.Fatal(err)
	}
	if err := b.ScalarOp(gpu.ScalarPow, y, z, 4, 2); err != nil {
		t.Fatal(err)
	}
	if err := b.MulScalar(z, y, 4, 2); err != nil {
		t.Fatal(err)
	}
	if err := b.Copy(y, z, 4); err != nil {
		t.Fatal(err)
	}
	want := []float32{8, 32, 72, 72}
	check(t, b, y, want)

	tmp := alloc(t, b, 4)
	before := submissionSerial(b)
	if err := b.Add(tmp, x, x, 4); err != nil {
		t.Fatal(err)
	}
	if err := b.Clamp(y, tmp, 4, -1, 6); err != nil {
		t.Fatal(err)
	}
	tmp.Close()
	if err := b.ScalarOp(gpu.ScalarPow, z, y, 4, 2); err != nil {
		t.Fatal(err)
	}
	if err := b.MulScalar(y, z, 4, 2); err != nil {
		t.Fatal(err)
	}
	if err := b.Copy(z, y, 4); err != nil {
		t.Fatal(err)
	}
	if submissionSerial(b) != before {
		t.Fatalf("flushed early before Download")
	}
	check(t, b, z, want)
	if submissionSerial(b)-before != 1 {
		t.Fatalf("got %d submissions for compute+copy+download, want 1", submissionSerial(b)-before)
	}

	// Descriptor chunk split when queued dispatches exceed batchHeapSize.
	b.mu.Lock()
	nd := b.driver.(*nativeDriver)
	origHeapSize := nd.batchHeapSize
	nd.batchHeapSize = 12
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		nd.batchHeapSize = origHeapSize
		b.mu.Unlock()
	}()
	before = submissionSerial(b)
	for i := 0; i < 6; i++ {
		if err := b.Add(y, x, x, 4); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if submissionSerial(b)-before < 2 {
		t.Fatalf("descriptor chunk split produced %d submissions, want >= 2", submissionSerial(b)-before)
	}
	check(t, b, y, []float32{2, 4, 6, 8})
}

func TestDirectMLRepeatedUncachedKeyCompiledOnce(t *testing.T) {
	b := testBackend(t)
	x := upload(t, b, []float32{1, 2, 3, 4})
	y := alloc(t, b, 4)
	z := alloc(t, b, 4)
	if err := b.lock(); err != nil {
		t.Fatal(err)
	}
	bx, err := b.native(x, 4)
	if err != nil {
		b.mu.Unlock()
		t.Fatal(err)
	}
	by, err := b.native(y, 4)
	if err != nil {
		b.mu.Unlock()
		t.Fatal(err)
	}
	bz, err := b.native(z, 4)
	if err != nil {
		b.mu.Unlock()
		t.Fatal(err)
	}
	steps := make([]dispatch, 0, maxCachedOperators+5)
	in := bx
	for i := 0; i < maxCachedOperators; i++ {
		out := by
		if i%2 == 1 {
			out = bz
		}
		steps = append(steps, dispatch{opKey{kind: opScalar, n: 4, scalar: math.Float32bits(1), bias: math.Float32bits(float32(i+1) * .125)}, out, []*allocation{in}})
		in = out
	}
	repeated := opKey{kind: opScalar, n: 4, scalar: math.Float32bits(1), bias: math.Float32bits(10)}
	for i := 0; i < 5; i++ {
		out := by
		if (maxCachedOperators+i)%2 == 1 {
			out = bz
		}
		steps = append(steps, dispatch{repeated, out, []*allocation{in}})
		in = out
	}
	b.driver.releaseOperators()
	nd := b.driver.(*nativeDriver)
	before := nd.serial
	if err := b.driver.executeBatch(steps); err != nil {
		b.mu.Unlock()
		t.Fatal(err)
	}
	if err := b.flushLocked(); err != nil {
		b.mu.Unlock()
		t.Fatal(err)
	}
	delta := nd.serial - before
	b.mu.Unlock()
	// 64 cached initializers + 1 uncached initializer (reused 5x) + 1 execution submit.
	if delta != maxCachedOperators+2 {
		t.Fatalf("submits=%d, want %d (repeated uncached key must compile once)", delta, maxCachedOperators+2)
	}
}
