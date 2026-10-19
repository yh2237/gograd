package vulkan

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func TestVulkanFusedBias(t *testing.T) {
	b := testBackend(t)
	for _, dims := range [][2]int{{0, 7}, {1, 1}, {2, 7}, {3, 129}, {74, 3072}} {
		t.Run(fmt.Sprint(dims), func(t *testing.T) {
			rows, cols := dims[0], dims[1]
			n := rows * cols
			x, bias, residual := make([]float32, n), make([]float32, cols), make([]float32, n)
			for i := range x {
				x[i] = float32(i%97-48) / 8
				residual[i] = float32(i%31-15) / 16
			}
			for i := range bias {
				bias[i] = float32(i%13-6) / 4
			}
			a, w, r, y := testUpload(t, b, x), testUpload(t, b, bias), testUpload(t, b, residual), testAlloc(t, b, n)
			want := make([]float32, n)
			before := b.MemoryStats()
			for repeat := 0; repeat < 3; repeat++ {
				if err := b.BiasGELU(y, a, w, rows, cols); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					v := float64(x[i] + bias[i%cols])
					want[i] = float32(.5 * v * (1 + math.Erf(v/math.Sqrt2)))
				}
				checkMatrix(t, b, y, want)
				if err := b.BiasResidual(y, a, w, r, rows, cols); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					want[i] = (x[i] + bias[i%cols]) + residual[i]
				}
				checkMatrix(t, b, y, want)
				// Inputs may alias; the output must be distinct.
				if err := b.BiasResidual(y, a, w, a, rows, cols); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					want[i] = (x[i] + bias[i%cols]) + x[i]
				}
				checkMatrix(t, b, y, want)
			}
			after := b.MemoryStats()
			if before.LiveBuffers != after.LiveBuffers || before.LiveBytes != after.LiveBytes {
				t.Fatal("fused call retained scratch", before, after)
			}
		})
	}
}

func TestVulkanFusedBiasOrder(t *testing.T) {
	b := testBackend(t)
	x := testUpload(t, b, []float32{1e20, 12, -12, -1e20})
	bias := testUpload(t, b, []float32{-1e20, -12, 12, 1e20})
	r := testUpload(t, b, []float32{1, 2, 3, 4})
	y := testAlloc(t, b, 4)
	if err := b.BiasResidual(y, x, bias, r, 1, 4); err != nil {
		t.Fatal(err)
	}
	checkValues(t, b, y, []float32{1, 2, 3, 4})
	// Clamp only after the bias addition, not according to the original x.
	if err := b.BiasGELU(y, x, bias, 1, 4); err != nil {
		t.Fatal(err)
	}
	checkValues(t, b, y, []float32{0, 0, 0, 0})
}

func TestVulkanFusedBiasPortable(t *testing.T) {
	d := &failingDriver{}
	b := NewBackend()
	b.once.Do(func() { b.driver = d })
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	x, w, r, y := testAlloc(t, b, 6), testAlloc(t, b, 3), testAlloc(t, b, 6), testAlloc(t, b, 6)
	for _, run := range []func() error{
		func() error { return b.BiasGELU(y, x, w, 2, 3) },
		func() error { return b.BiasResidual(y, x, w, r, 2, 3) },
	} {
		before := b.MemoryStats()
		calls := d.computeCalls
		if err := run(); err != nil {
			t.Fatal(err)
		}
		if d.computeCalls != calls+1 || b.MemoryStats() != before {
			t.Fatal("fusion did not dispatch once without scratch")
		}
	}
	for _, err := range []error{
		b.BiasGELU(y, x, w, -1, 3), b.BiasGELU(y, x, w, 2, 0), b.BiasGELU(y, x, w, math.MaxInt32, 2),
		b.BiasGELU(y, x, w, 1, 3), b.BiasGELU(x, x, w, 2, 3), b.BiasGELU(w, w, w, 1, 3),
		b.BiasResidual(r, x, w, r, 2, 3),
	} {
		if !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
	}
	if d.computeCalls != 2 || !b.Available() {
		t.Fatal("invalid arguments dispatched or disabled backend")
	}
	if err := b.BiasResidual(y, x, w, nil, 2, 3); !errors.Is(err, gpu.ErrClosed) {
		t.Fatal(err)
	}
	other := NewBackend()
	other.once.Do(func() { other.driver = &failingDriver{} })
	foreign := testAlloc(t, other, 3)
	defer func() { foreign.Close(); other.Close() }()
	if err := b.BiasGELU(y, x, foreign, 2, 3); !errors.Is(err, gpu.ErrBackendMismatch) {
		t.Fatal(err)
	}
	zeroX, zeroY := testAlloc(t, b, 0), testAlloc(t, b, 0)
	if err := b.BiasGELU(zeroY, zeroX, w, 0, 3); err != nil {
		t.Fatal(err)
	}
	if err := b.BiasResidual(zeroY, zeroX, w, zeroX, 0, 3); err != nil {
		t.Fatal(err)
	}
	if d.computeCalls != 2 {
		t.Fatal("empty operation dispatched")
	}
	w.Close()
	if err := b.BiasGELU(y, x, w, 2, 3); !errors.Is(err, gpu.ErrClosed) {
		t.Fatal(err)
	}
}

func TestVulkanFusedBiasFailures(t *testing.T) {
	for _, residual := range []bool{false, true} {
		t.Run(fmt.Sprint(residual), func(t *testing.T) {
			d := &failingDriver{failure: errors.New("injected fused dispatch failure")}
			b := NewBackend()
			b.once.Do(func() { b.driver = d })
			x, w, y := testAlloc(t, b, 6), testAlloc(t, b, 3), testAlloc(t, b, 6)
			var err error
			if residual {
				err = b.BiasResidual(y, x, w, x, 2, 3)
			} else {
				err = b.BiasGELU(y, x, w, 2, 3)
			}
			if !errors.Is(err, d.failure) || b.Available() || d.computeCalls != 1 {
				t.Fatal(err)
			}
			x.Close()
			w.Close()
			y.Close()
			if d.allocated != d.freed || b.MemoryStats().ReservedBytes != 0 {
				t.Fatal("fused failure leaked resources")
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
