package autograd

import (
	"github.com/yh2237/gograd/gpu"
	"testing"
)

// Disabled by default so the original driver-free tests continue exercising
// the materialization fallback when a backend rejects a particular layout.
func (b *inferenceTestBackend) AddViews(out *gpu.Buffer, a, right gpu.View) error {
	return b.binaryViews(out, a, right, false)
}
func (b *inferenceTestBackend) MulViews(out *gpu.Buffer, a, right gpu.View) error {
	return b.binaryViews(out, a, right, true)
}
func (b *inferenceTestBackend) binaryViews(out *gpu.Buffer, a, right gpu.View, multiply bool) error {
	b.mu.Lock()
	enabled := b.viewBinaryEnabled
	fail := b.failNext
	if enabled {
		b.failNext = false
	}
	b.mu.Unlock()
	if !enabled {
		return gpu.ErrUnsupported
	}
	n, err := a.Validate(b)
	if err != nil {
		return err
	}
	if _, err := right.Validate(b); err != nil {
		return err
	}
	x, err := b.memory(a.Buffer, a.Buffer.Len())
	if err != nil {
		return err
	}
	w, err := b.memory(right.Buffer, right.Buffer.Len())
	if err != nil {
		return err
	}
	y, err := b.memory(out, n)
	if err != nil {
		return err
	}
	if fail {
		return injectedInferenceError
	}
	for i := range y.data {
		u, v := x.data[storageIndex(i, a.Shape, a.Strides, a.Offset)], w.data[storageIndex(i, right.Shape, right.Strides, right.Offset)]
		if multiply {
			y.data[i] = u * v
		} else {
			y.data[i] = u + v
		}
	}
	return nil
}

func TestInferenceGPUViewBinaryCapability(t *testing.T) {
	x := inferenceTestNew(t, fixtureDevice, []int{1, 3}, 1)
	bias := inferenceTestNew(t, fixtureDevice, []int{3}, 2)
	v := Expand(x, 2, 3)
	defer v.ReleaseGraph()
	backend := inferenceFixture
	defer func() {
		backend.mu.Lock()
		backend.viewBinaryEnabled = false
		backend.failNext = false
		backend.mu.Unlock()
	}()
	for _, enabled := range []bool{false, true} {
		backend.mu.Lock()
		backend.viewBinaryEnabled = enabled
		backend.mu.Unlock()
		_, allocs, downloads := backend.snapshot()
		y := Add(v, bias)
		_, after, reads := backend.snapshot()
		want := 3
		if enabled {
			want = 1
		}
		if after-allocs != want || reads != downloads {
			t.Fatalf("enabled=%t: allocations=%d reads=%d", enabled, after-allocs, reads-downloads)
		}
		got, err := y.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		xh, err := x.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		bh, err := bias.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		for i, value := range got {
			if value != xh[i%3]+bh[i%3] {
				t.Fatal("binary view agreement")
			}
		}
		y.Close()
	}
	live, _, _ := backend.snapshot()
	backend.mu.Lock()
	backend.failNext = true
	backend.mu.Unlock()
	assertGPUErrorPanic(t, injectedInferenceError, func() { Mul(v, bias) })
	if after, _, _ := backend.snapshot(); after != live {
		t.Fatal("view binary failure leaked output")
	}
}
