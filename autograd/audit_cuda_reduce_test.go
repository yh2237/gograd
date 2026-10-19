package autograd

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestAuditScalarReductionsCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	for _, n := range []int{1, 31, 255, 256, 257, 65537, 1 << 20} {
		for _, nd := range []bool{false, true} {
			for _, mean := range []bool{false, true} {
				t.Run(fmt.Sprintf("n=%d/nd=%t/mean=%t", n, nd, mean), func(t *testing.T) {
					shape, axes := []int{n}, []int{0}
					if nd {
						shape, axes = []int{1, 1, 1, 1, n}, []int{0, 1, 2, 3, 4}
					}
					values := make([]float32, n)
					var sum float32
					for i := range values {
						values[i] = float32(i%7-3) * .25
						sum += values[i]
					}
					x := conv2dTestTensor(t, values, shape, tensor.CUDA, true)
					var y *Tensor
					grad := float32(1)
					if mean {
						y, sum, grad = Mean(x, axes...), sum/float32(n), 1/float32(n)
					} else {
						y = Sum(x, axes...)
					}
					defer y.ReleaseGraph()
					conv2dClose(t, "scalar reduction", conv2dHost(t, y, false), []float32{sum}, 2e-6, 2e-6)
					if err := y.Backward(); err != nil {
						t.Fatal(err)
					}
					got := conv2dHost(t, x, true)
					for i, g := range got {
						if g != grad {
							t.Fatalf("gradient[%d]=%g, want %g", i, g, grad)
						}
					}
				})
			}
		}
	}
}

func TestAuditScalarReductionViewsAndPartialCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	x := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{3, 4}, tensor.CUDA, true)
	view := Slice(Transpose(x, 0, 1), 0, 1, 3)
	full := Mean(view, 0, 1)
	defer full.ReleaseGraph()
	conv2dClose(t, "view mean", conv2dHost(t, full, false), []float32{6.5}, 1e-6, 1e-6)
	if err := full.Backward(); err != nil {
		t.Fatal(err)
	}
	g := float32(1.0 / 6)
	conv2dClose(t, "view gradient", conv2dHost(t, x, true), []float32{0, g, g, 0, 0, g, g, 0, 0, g, g, 0}, 1e-6, 1e-6)
	// Partial reduction must not use the scalar-only output mapping.
	partial := Sum(x, 0)
	defer partial.ReleaseGraph()
	conv2dClose(t, "partial sum", conv2dHost(t, partial, false), []float32{15, 18, 21, 24}, 1e-6, 1e-6)
	x.ZeroGrad()
	loss := Sum(partial, 0)
	defer loss.ReleaseGraph()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "partial gradient", conv2dHost(t, x, true), []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1e-6, 1e-6)
}

func TestAuditScalarReductionCaptureCUDA(t *testing.T) {
	ctx := auditCUDAContext(t)
	defer ctx.Close()
	const n = 65537
	values := make([]float32, n)
	for i := range values {
		values[i] = 2
	}
	x := conv2dTestTensor(t, values, []int{n}, tensor.CUDA, false)
	warm := Mean(x, 0)
	warm.ReleaseGraph()
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	var y *Tensor
	graph, err := cuda.Capture(stream, func() error { y = Mean(x, 0); return nil })
	if err != nil {
		t.Fatal(err)
	}
	defer y.ReleaseGraph()
	defer graph.Close()
	for _, value := range []float32{2, -3, .25} {
		for i := range values {
			values[i] = value
		}
		if err := x.CopyFrom(values); err != nil {
			t.Fatal(err)
		}
		if err := graph.Launch(); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "captured mean", conv2dHost(t, y, false), []float32{value}, 3e-6, 3e-6)
	}
}

// Baseline and block versions include identical zeroing/launch overhead. Run
// serially with CUDA_PATH set; no timings are asserted by the regression suite.
func BenchmarkAuditScalarReduceCUDA(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA unavailable")
	}
	for _, n := range []int{256, 65536, 1 << 20} {
		for _, mean := range []bool{false, true} {
			for _, baseline := range []bool{true, false} {
				b.Run(fmt.Sprintf("n=%d/mean=%t/elementAtomics=%t", n, mean, baseline), func(b *testing.B) {
					benchmarkAuditScalarReduce(b, n, mean, baseline)
				})
			}
		}
	}
}

// benchmarkAuditScalarReduce measures one configuration. testing runs every
// sub-benchmark on a new goroutine, and a CUDA context belongs to the OS
// thread that created it, so the context has to be created here instead of in
// the parent benchmark.
func benchmarkAuditScalarReduce(b *testing.B, n int, mean, baseline bool) {
	ctx, err := NewCUDAContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Close()
	x, y := mustAlloc(n), mustAlloc(1)
	defer x.Free()
	defer y.Free()
	if err := x.Memset(0, x.Size()); err != nil {
		b.Fatal(err)
	}
	graphKernel("reduce_scalar_f") // Compile before timing.
	xp, yp, size := ptr(x), ptr(y), int32(n)
	d1, d2, zero := int32(1), int32(n), int32(0)
	scale := float32(1)
	if mean {
		scale /= float32(n)
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(n * 4))
	b.ResetTimer()
	for range b.N {
		if err := y.Memset(0, 4); err != nil {
			b.Fatal(err)
		}
		if baseline {
			launch("reduce_f", n, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&size), unsafe.Pointer(&d1), unsafe.Pointer(&d2), unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&scale))
		} else {
			launchScalarReduce(xp, yp, size, scale)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

// testing runs every sub-benchmark on its own goroutine. A CUDA context is
// current only on the OS thread that created it, so a benchmark body that
// launches a kernel has to create the context in the sub-benchmark itself.
// Inheriting the parent's context fails with "invalid context".
func TestAuditScalarReduceBenchmarkContextPerSubBenchmark(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	result := testing.Benchmark(func(b *testing.B) {
		b.Run("sub", func(b *testing.B) {
			benchmarkAuditScalarReduce(b, 256, false, false)
		})
	})
	if result.N == 0 {
		t.Fatal("scalar reduce benchmark did not run")
	}
}
