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
	computeCalls, failAt     int
}

func (*failingDriver) info() Info {
	return Info{MaxStorageBufferBytes: 1 << 30, MaxWorkGroupCount: [3]uint32{65535, 65535, 65535}}
}
func (d *failingDriver) alloc(size uint64, host bool) (*block, error) {
	d.allocated++
	return &block{size: size, allocated: size, host: host}, nil
}
func (d *failingDriver) free(*block) bool                     { d.freed++; return true }
func (*failingDriver) write(*block, []float32)                {}
func (*failingDriver) read(*block, []float32)                 {}
func (d *failingDriver) copy(out, src *block, n uint64) error { return d.failure }
func (d *failingDriver) zero(*block, uint64) error            { return d.failure }
func (d *failingDriver) pointwise(out, a, b *block, n, kind uint32, scalar float32) error {
	return d.failure
}
func (d *failingDriver) fusedBias(out, a, bias, residual *block, n, channels, kind uint32) error {
	return d.computeFailure()
}
func (d *failingDriver) close() error { d.closed++; return nil }
func (d *failingDriver) reduce(out, a, weight, bias *block, count, affineCount uint32, p reduceParams) error {
	return d.computeFailure()
}
func (d *failingDriver) conv1d(p convBatch) error {
	stages := 2
	if p.hasBias {
		stages++
	}
	for i := 0; i < stages; i++ {
		if err := d.computeFailure(); err != nil {
			return err
		}
	}
	return nil
}
func (d *failingDriver) viewCopy(out, src *block, outCount, srcCount uint32, p viewParams) error {
	return d.failure
}
func (d *failingDriver) binaryViews(out, a, b *block, outCount, aCount, bCount uint32, p binaryParams) error {
	return d.failure
}
func (d *failingDriver) gemm(out, a, b *block, outCount, aCount, bCount uint32, p gemmParams, variant int) error {
	return d.computeFailure()
}

func (d *failingDriver) computeFailure() error {
	d.computeCalls++
	if d.failAt == 0 || d.computeCalls == d.failAt {
		return d.failure
	}
	return nil
}

func TestVulkanTransformerFailures(t *testing.T) {
	for _, operation := range []string{"softmax", "layernorm", "groupnorm", "conv_im2col", "conv_gemm", "conv_bias"} {
		t.Run(operation, func(t *testing.T) {
			d := &failingDriver{failure: errors.New("injected transformer failure"), failAt: 1}
			if operation == "conv_gemm" {
				d.failAt = 2
			}
			if operation == "conv_bias" {
				d.failAt = 3
			}
			b := NewBackend()
			b.once.Do(func() { b.driver = d })
			t.Cleanup(func() {
				if err := b.Close(); err != nil {
					t.Error(err)
				}
			})
			var buffers []*gpu.Buffer
			alloc := func(n int) *gpu.Buffer {
				v, err := b.Alloc(n)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(v.Close)
				buffers = append(buffers, v)
				return v
			}
			x, w, bias, y := alloc(8), alloc(4), alloc(4), alloc(8)
			var err error
			switch operation {
			case "softmax":
				err = b.Softmax(y, x, 2, 4)
			case "layernorm":
				err = b.LayerNorm(y, x, w, bias, 2, 4, 1e-5)
			case "groupnorm":
				err = b.GroupNorm(y, x, w, bias, 1, 2, 4, 2, 1e-5)
			default:
				err = b.Conv1d(y, x, alloc(16), bias, gpu.Conv1dOptions{Batch: 1, InputLength: 2, InChannels: 4, OutChannels: 4, Kernel: 1, Stride: 1, Dilation: 1})
			}
			if !errors.Is(err, d.failure) || b.Available() {
				t.Fatalf("%v, available=%t", err, b.Available())
			}
			for _, v := range buffers {
				v.Close()
			}
			if s := b.MemoryStats(); s.LiveBuffers != 0 || s.ReservedBytes != 0 {
				t.Fatalf("failed compute leaked scratch: %+v", s)
			}
		})
	}
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
