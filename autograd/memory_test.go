package autograd

import (
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// Exercises the model/optimizer lifetime used by variable-length TCN trainers.
func TestVariableLengthCUDAMemory(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, e := NewCUDAContext()
	if e != nil {
		t.Fatal(e)
	}
	defer ctx.Close()
	if e = cuda.ReleasePool(); e != nil {
		t.Fatal(e)
	}
	m, e := NewSpeechTiming(4, tensor.CUDA, 17)
	if e != nil {
		t.Fatal(e)
	}
	params := m.Parameters()
	opt := NewAdamW(params, .001, .0001)
	base := cuda.MemoryStats()
	cuda.ResetAllocationPeak()
	runStep := func(step int) cuda.AllocationStats {
		length := 96 + (step*37)%300
		ids := make([]int, 8*length*3)
		for i := range ids {
			ids[i] = 3 + (i+step)%37
		}
		cont, e := Zeros([]int{8, length, 4}, tensor.CUDA, false)
		if e != nil {
			t.Fatal(e)
		}
		target, e := Zeros([]int{8, length, 80}, tensor.CUDA, false)
		if e != nil {
			t.Fatal(e)
		}
		opt.ZeroGrad()
		loss := MaskedLoss(m.Forward(ids, cont, uint32(step*8)), target, false)
		if e = loss.Backward(); e != nil {
			t.Fatal(e)
		}
		ClipGradNorm(params, 1)
		opt.Step()
		loss.ReleaseGraph()
		cont.Close()
		target.Close()
		s := cuda.MemoryStats()
		if s.CachedBytes > s.CacheLimitBytes {
			t.Fatalf("step %d: cached %d exceeds cap %d", step+1, s.CachedBytes, s.CacheLimitBytes)
		}
		if s.ReservedBytes != s.LiveBytes+s.CachedBytes {
			t.Fatalf("step %d: inconsistent pool accounting: %+v", step+1, s)
		}
		if s.LiveBytes > base.LiveBytes+(4<<20) {
			t.Fatalf("step %d: live bytes grew from %d to %d", step+1, base.LiveBytes, s.LiveBytes)
		}
		return s
	}
	const steps = 200
	for step := 0; step < steps; step++ {
		s := runStep(step)
		length := 96 + (step*37)%300
		if step == 0 || step == 31 || step == 63 || step == 127 || step == steps-1 {
			t.Logf("step=%d length=%d live=%d cached=%d reserved=%d peak_reserved=%d", step+1, length, s.LiveBytes, s.CachedBytes, s.ReservedBytes, s.PeakReservedBytes)
		}
	}
	if e = cuda.Synchronize(); e != nil {
		t.Fatal(e)
	}
	end := cuda.MemoryStats()
	t.Logf("baseline live=%d cached=%d reserved=%d; end live=%d cached=%d reserved=%d peak_reserved=%d", base.LiveBytes, base.CachedBytes, base.ReservedBytes, end.LiveBytes, end.CachedBytes, end.ReservedBytes, end.PeakReservedBytes)
	if e = cuda.SetPoolCacheLimit(256 << 20); e != nil {
		t.Fatal(e)
	}
	defer cuda.SetPoolCacheLimit(base.CacheLimitBytes)
	trimmed := cuda.MemoryStats()
	if trimmed.CachedBytes > 256<<20 {
		t.Fatalf("trimmed cache %d exceeds 256 MiB", trimmed.CachedBytes)
	}
	t.Logf("trimmed cached=%d reserved=%d", trimmed.CachedBytes, trimmed.ReservedBytes)
	for step := steps; step < steps+64; step++ {
		runStep(step)
	}
	bounded := cuda.MemoryStats()
	t.Logf("bounded phase live=%d cached=%d reserved=%d peak_reserved=%d", bounded.LiveBytes, bounded.CachedBytes, bounded.ReservedBytes, bounded.PeakReservedBytes)
}
