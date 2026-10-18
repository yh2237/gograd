package directml

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sync"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func testBackend(t *testing.T) *Backend {
	t.Helper()
	b := NewBackend()
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	if !b.Available() {
		if os.Getenv("GOGRAD_REQUIRE_DIRECTML") == "1" {
			t.Fatal(b.Err())
		}
		t.Skipf("DirectML unavailable: %v", b.Err())
	}
	return b
}
func alloc(t *testing.T, b *Backend, n int) *gpu.Buffer {
	t.Helper()
	v, err := b.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(v.Close)
	return v
}
func upload(t *testing.T, b *Backend, data []float32) *gpu.Buffer {
	t.Helper()
	v, err := b.Upload(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(v.Close)
	return v
}
func check(t *testing.T, b *Backend, out *gpu.Buffer, want []float32) {
	t.Helper()
	got, err := b.Download(out, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want[i])) > 1e-5+1e-5*math.Abs(float64(want[i])) {
			t.Fatalf("[%d]=%g want %g", i, v, want[i])
		}
	}
}

func TestDirectMLDevice(t *testing.T) {
	b := testBackend(t)
	info, err := b.DeviceInfo()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("DirectML GPU: %+v", info)
	if info.Name == "" || info.FeatureLevel < 0x1000 {
		t.Fatal(info)
	}
}

func TestDirectMLPointwise(t *testing.T) {
	b := testBackend(t)
	for _, n := range []int{0, 1, 129, 100003} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			a, right, want := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range a {
				a[i] = float32(i%37-18) / 16
				right[i] = float32(i%13-6) / 8
			}
			x, w, y := upload(t, b, a), upload(t, b, right), alloc(t, b, n)
			if err := b.Add(y, x, w, n); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] = a[i] + right[i]
			}
			check(t, b, y, want)
			if err := b.Mul(y, x, w, n); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] = a[i] * right[i]
			}
			check(t, b, y, want)
			if err := b.MulScalar(y, x, n, -.75); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] = a[i] * -.75
			}
			check(t, b, y, want)
			if err := b.Zero(y); err != nil {
				t.Fatal(err)
			}
			clear(want)
			check(t, b, y, want)
			if err := b.Copy(y, x, n); err != nil {
				t.Fatal(err)
			}
			check(t, b, y, a)
		})
	}
}

func TestDirectMLGEMM(t *testing.T) {
	b := testBackend(t)
	rng := rand.New(rand.NewPCG(19, 73))
	for _, dims := range [][3]int{{1, 1, 1}, {7, 11, 13}, {17, 19, 33}, {0, 5, 3}, {5, 0, 3}, {3, 5, 0}, {74, 768, 768}} {
		for _, ta := range []bool{false, true} {
			for _, tb := range []bool{false, true} {
				t.Run(fmt.Sprintf("%v/%t/%t", dims, ta, tb), func(t *testing.T) {
					m, n, k := dims[0], dims[1], dims[2]
					a, right, want := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
					for i := range a {
						a[i] = rng.Float32() - .5
					}
					for i := range right {
						right[i] = rng.Float32() - .5
					}
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							var sum float32
							for d := 0; d < k; d++ {
								ai, bi := i*k+d, d*n+j
								if ta {
									ai = d*m + i
								}
								if tb {
									bi = j*k + d
								}
								sum += a[ai] * right[bi]
							}
							want[i*n+j] = sum
						}
					}
					out := alloc(t, b, m*n)
					if err := b.MatMul(out, upload(t, b, a), upload(t, b, right), m, n, k, ta, tb); err != nil {
						t.Fatal(err)
					}
					check(t, b, out, want)
				})
			}
		}
	}
}

func TestDirectMLValidation(t *testing.T) {
	b := testBackend(t)
	x, y := alloc(t, b, 4), alloc(t, b, 4)
	for _, err := range []error{b.Add(x, x, y, 4), b.Mul(y, x, x, 3), b.MatMul(y, x, x, -1, 2, 2, false, false)} {
		if !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
	}
	if err := b.GELU(y, x, 4); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatal(err)
	}
	x.Close()
	if err := b.MulScalar(y, x, 4, 1); !errors.Is(err, gpu.ErrClosed) {
		t.Fatal(err)
	}
	if !b.Available() {
		t.Fatal("validation disabled healthy device")
	}
}

func TestDirectMLCacheAndPool(t *testing.T) {
	b := testBackend(t)
	b.SetPoolCacheLimit(1024)
	x, y := upload(t, b, []float32{1, 2, 3, 4}), alloc(t, b, 4)
	for i := 0; i < 70; i++ {
		if err := b.MulScalar(y, x, 4, float32(i)); err != nil {
			t.Fatal(err)
		}
	}
	stats := b.OperatorStats()
	if stats.CachedOperators < 1 || stats.CachedOperators > 64 || stats.PersistentBytes+stats.TemporaryBytes > 256<<20 {
		t.Fatal(stats)
	}
	check(t, b, y, []float32{69, 138, 207, 276})
	b.ReleaseOperators()
	if stats := b.OperatorStats(); stats != (OperatorStats{}) {
		t.Fatal(stats)
	}
	if err := b.Add(y, x, x, 4); err != nil {
		t.Fatal(err)
	}
	check(t, b, y, []float32{2, 4, 6, 8})
	x.Close()
	y.Close()
	if stats := b.MemoryStats(); stats.LiveBuffers != 0 || stats.LiveBytes != 0 || stats.CachedBytes > 1024 {
		t.Fatal(stats)
	}
	b.ReleasePool()
	if stats := b.MemoryStats(); stats.ReservedBytes != 0 {
		t.Fatal(stats)
	}
	b.ReleaseOperators()
}

func TestDirectMLConcurrent(t *testing.T) {
	b := testBackend(t)
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for task := 0; task < 8; task++ {
		wg.Add(1)
		go func(task int) {
			defer wg.Done()
			n := 65 + task
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i)
			}
			x, err := b.Upload(data)
			if err != nil {
				errs <- err
				return
			}
			defer x.Close()
			y, err := b.Alloc(n)
			if err != nil {
				errs <- err
				return
			}
			defer y.Close()
			for i := 0; i < 5; i++ {
				if err := b.MulScalar(y, x, n, 2); err != nil {
					errs <- err
					return
				}
				values, err := b.Download(y, n)
				if err != nil {
					errs <- err
					return
				}
				for j, v := range values {
					if v != 2*data[j] {
						errs <- fmt.Errorf("task %d index %d", task, j)
						return
					}
				}
			}
		}(task)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if stats := b.MemoryStats(); stats.LiveBuffers != 0 {
		t.Fatal(stats)
	}
}
