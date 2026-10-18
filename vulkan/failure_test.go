package vulkan

import (
	"errors"
	"github.com/yh2237/gograd/gpu"
	"testing"
)

// Driver failure/cleanup tests do not require a loader, GPU or shader compiler.
type failingDriver struct {
	failure                  error
	allocated, freed, closed int
}

func (*failingDriver) info() Info {
	return Info{MaxStorageBufferBytes: 1 << 30, MaxWorkGroupCount: [3]uint32{65535, 65535, 65535}}
}
func (d *failingDriver) alloc(size uint64, host bool) (*block, error) {
	d.allocated++
	return &block{size: size, allocated: size, host: host}, nil
}
func (d *failingDriver) free(*block)                          { d.freed++ }
func (*failingDriver) write(*block, []float32)                {}
func (*failingDriver) read(*block, []float32)                 {}
func (d *failingDriver) copy(out, src *block, n uint64) error { return d.failure }
func (d *failingDriver) zero(*block, uint64) error            { return d.failure }
func (d *failingDriver) pointwise(out, a, b *block, n, kind uint32, scalar float32) error {
	return d.failure
}
func (d *failingDriver) close() { d.closed++ }
func (d *failingDriver) viewCopy(out, src *block, outCount, srcCount uint32, p viewParams) error {
	return d.failure
}
func (d *failingDriver) gemm(out, a, b *block, outCount, aCount, bCount uint32, p gemmParams, variant int) error {
	return d.failure
}

func TestVulkanFailureDisablesBackend(t *testing.T) {
	d := &failingDriver{failure: errors.New("injected queue failure")}
	b := NewBackend()
	b.once.Do(func() { b.driver = d })
	x, err := b.Alloc(3)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.CopyFrom(x, []float32{1, 2, 3}); !errors.Is(err, d.failure) {
		t.Fatal(err)
	}
	if b.Available() || b.Err() == nil {
		t.Fatal("failed device advertised as available")
	}
	x.Close()
	x.Close()
	if s := b.MemoryStats(); s.LiveBytes != 0 || s.ReservedBytes != 0 || s.LiveBuffers != 0 {
		t.Fatalf("%+v", s)
	}
	if d.allocated != d.freed {
		t.Fatalf("%d allocated, %d freed", d.allocated, d.freed)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if d.closed != 1 {
		t.Fatal("driver not closed")
	}
}

func TestVulkanViewAndGEMMFailures(t *testing.T) {
	for _, operation := range []string{"gather", "scatter", "gemm"} {
		t.Run(operation, func(t *testing.T) {
			d := &failingDriver{failure: errors.New("injected compute failure")}
			b := NewBackend()
			b.once.Do(func() { b.driver = d })
			buffers := make([]*gpu.Buffer, 3)
			for i := range buffers {
				var err error
				buffers[i], err = b.Alloc(4)
				if err != nil {
					t.Fatal(err)
				}
			}
			v := gpu.View{Buffer: buffers[0], Shape: []int{2, 2}, Strides: []int{2, 1}}
			var err error
			switch operation {
			case "gather":
				err = b.Gather(buffers[2], v)
			case "scatter":
				err = b.Scatter(v, buffers[2])
			case "gemm":
				err = b.MatMul(buffers[2], buffers[0], buffers[1], 2, 2, 2, false, false)
			}
			if !errors.Is(err, d.failure) || b.Available() {
				t.Fatalf("%v, available=%t", err, b.Available())
			}
			for _, buffer := range buffers {
				buffer.Close()
			}
			if s := b.MemoryStats(); s.LiveBuffers != 0 || s.ReservedBytes != 0 {
				t.Fatalf("%+v", s)
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDispatchGrid(t *testing.T) {
	limits := [3]uint32{2, 3, 1}
	for _, tc := range []struct{ n, x, y uint32 }{{1, 1, 1}, {256, 2, 1}, {257, 2, 2}, {513, 2, 3}} {
		x, y, err := dispatchGrid(tc.n, limits)
		if err != nil || x != tc.x || y != tc.y {
			t.Fatalf("%d: %d,%d %v", tc.n, x, y, err)
		}
	}
	if _, _, err := dispatchGrid(769, limits); err == nil {
		t.Fatal("oversized dispatch accepted")
	}
	if _, _, err := dispatchGrid(1, [3]uint32{}); err == nil {
		t.Fatal("empty group limit accepted")
	}
}

func TestVulkanRegistration(t *testing.T) {
	backend, ok := gpu.Lookup("vulkan")
	if !ok || backend != defaultBackend {
		t.Fatal("default Vulkan backend not registered")
	}
	if err := defaultBackend.Close(); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatal(err)
	}
	// Registration and rejecting Close do not initialize a native driver.
	if stats := defaultBackend.MemoryStats(); stats.ReservedBytes != 0 {
		t.Fatalf("unexpected implicit allocation: %+v", stats)
	}
}
