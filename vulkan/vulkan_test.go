package vulkan

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func testBackend(t *testing.T) *Backend {
	t.Helper()
	b := NewBackend()
	if !b.Available() {
		err := b.Err()
		b.Close()
		if os.Getenv("GOGRAD_REQUIRE_VULKAN") == "1" {
			t.Fatal(err)
		}
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	return b
}

func testUpload(t *testing.T, b *Backend, values []float32) *gpu.Buffer {
	t.Helper()
	x, err := b.Upload(values)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(x.Close)
	return x
}
func testAlloc(t *testing.T, b *Backend, n int) *gpu.Buffer {
	t.Helper()
	x, err := b.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(x.Close)
	return x
}
func checkValues(t *testing.T, b *Backend, x *gpu.Buffer, want []float32) {
	t.Helper()
	got, err := b.Download(x, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.Abs(float64(value-want[i])) > 1e-5 {
			t.Fatalf("[%d]=%g want %g", i, value, want[i])
		}
	}
}

func TestEmbeddedShader(t *testing.T) {
	for _, code := range [][]byte{pointwiseSPV, viewSPV, gemmSPV, reduceSPV, convSPV} {
		if len(code) < 20 || len(code)%4 != 0 || binary.LittleEndian.Uint32(code) != 0x07230203 {
			t.Fatal("invalid embedded SPIR-V")
		}
	}
}

func TestVulkanPointwise(t *testing.T) {
	b := testBackend(t)
	info, err := b.DeviceInfo()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GPU %s, Vulkan %d.%d.%d, compute queue %d, max storage %d bytes", info.Name, info.APIVersion>>22, info.APIVersion>>12&1023, info.APIVersion&4095, info.QueueFamily, info.MaxStorageBufferBytes)
	for _, n := range []int{0, 1, 127, 128, 129, 513, 100003} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			a, right, want := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range a {
				a[i] = float32(i%97-48) / 16
				right[i] = float32(i%37-18) / 32
			}
			x, y, out := testUpload(t, b, a), testUpload(t, b, right), testAlloc(t, b, n)
			if err := b.Add(out, x, y, n); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] = a[i] + right[i]
			}
			checkValues(t, b, out, want)
			if err := b.Mul(out, x, y, n); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] = a[i] * right[i]
			}
			checkValues(t, b, out, want)
			if err := b.MulScalar(out, x, n, .375); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] = a[i] * .375
			}
			checkValues(t, b, out, want)
			if err := b.Copy(out, x, n); err != nil {
				t.Fatal(err)
			}
			checkValues(t, b, out, a)
			if err := b.Zero(out); err != nil {
				t.Fatal(err)
			}
			clear(want)
			checkValues(t, b, out, want)
		})
	}
	if stats := b.MemoryStats(); stats.LiveBuffers != 0 || stats.LiveBytes != 0 {
		t.Fatalf("live memory leaked: %+v", stats)
	}
	b.ReleasePool()
	if b.MemoryStats().ReservedBytes != 0 {
		t.Fatal("pool memory not returned")
	}
}

func TestVulkanValidationAndPool(t *testing.T) {
	b := testBackend(t)
	b.SetPoolCacheLimit(4096)
	x := testUpload(t, b, []float32{1, 2, 3})
	y := testAlloc(t, b, 3)
	if err := b.Close(); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.CopyFrom(x, []float32{4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	checkValues(t, b, x, []float32{4, 5, 6})
	if prefix, err := b.Download(x, 2); err != nil || len(prefix) != 2 || prefix[1] != 5 {
		t.Fatalf("%v %v", prefix, err)
	}
	if _, err := b.Download(x, 4); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.CopyFrom(x, []float32{1}); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.Add(x, x, x, 3); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.Add(y, x, x, 2); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := b.Alloc(-1); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.MatMul(y, x, x, 1, 3, 3, false, false); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if !b.Available() {
		t.Fatal("validation error disabled a healthy device")
	}
	x.Close()
	x.Close()
	y.Close()
	if err := b.Zero(x); !errors.Is(err, gpu.ErrClosed) {
		t.Fatal(err)
	}
	for _, n := range []int{1, 127, 513, 4097, 3} {
		buffer, err := b.Upload(make([]float32, n))
		if err != nil {
			t.Fatal(err)
		}
		buffer.Close()
		if s := b.MemoryStats(); s.LiveBuffers != 0 || s.CachedBytes > s.CacheLimitBytes {
			t.Fatalf("unbounded pool: %+v", s)
		}
	}
	b.SetPoolCacheLimit(0)
	if stats := b.MemoryStats(); stats.ReservedBytes != 0 {
		t.Fatalf("%+v", stats)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Alloc(1); !errors.Is(err, gpu.ErrClosed) {
		t.Fatal(err)
	}
}

func TestVulkanConcurrent(t *testing.T) {
	b := testBackend(t)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for pass := 0; pass < 10; pass++ {
				x, err := b.Upload([]float32{float32(w), 2, 3})
				if err != nil {
					t.Error(err)
					return
				}
				y, err := b.Alloc(3)
				if err != nil {
					x.Close()
					t.Error(err)
					return
				}
				err = b.MulScalar(y, x, 3, 2)
				if err == nil {
					var got []float32
					got, err = b.Download(y, 3)
					if err == nil && (got[0] != float32(w)*2 || got[1] != 4 || got[2] != 6) {
						t.Errorf("worker %d: %v", w, got)
					}
				}
				y.Close()
				x.Close()
				if err != nil {
					t.Error(err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	if s := b.MemoryStats(); s.LiveBuffers != 0 || s.LiveBytes != 0 {
		t.Fatalf("%+v", s)
	}
}
