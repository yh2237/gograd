package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"sync"
)

// cuBLAS handles are not safe to share between host threads, and the driver
// scopes the current context per thread. Keep one handle per OS thread and
// device instead of one process-wide handle, so independent CUDA training runs
// do not serialize on or corrupt a shared handle.
//
// The handle follows the calling thread's implicit stream: kernels, memset and
// copies already do, so a GEMM issued without naming a stream must land on the
// same stream. The selected stream is cached to avoid a driver call per GEMM.
type threadBLAS struct {
	handle *cuda.Blas
	device int
	stream *cuda.Stream
}

var blasHandles sync.Map // uintptr -> *threadBLAS

func blas() *cuda.Blas {
	device, err := cuda.CurrentDevice()
	if err != nil {
		panic(err)
	}
	stream := cuda.CurrentStream()
	id := cuda.ThreadID()
	if value, ok := blasHandles.Load(id); ok {
		entry := value.(*threadBLAS)
		if entry.device != device {
			// The Go runtime reuses OS threads. A thread that now drives another
			// device must not keep a handle bound to the previous context.
			_ = entry.handle.Destroy()
		} else if entry.stream == stream {
			return entry.handle
		} else {
			if err := entry.handle.SetStream(stream); err != nil {
				panic(err)
			}
			entry.stream = stream
			return entry.handle
		}
	}
	handle, err := cuda.NewBlas()
	if err != nil {
		panic(err)
	}
	if stream != nil {
		if err := handle.SetStream(stream); err != nil {
			panic(err)
		}
	}
	blasHandles.Store(id, &threadBLAS{handle: handle, device: device, stream: stream})
	return handle
}

// SetBLASStream binds autograd's cuBLAS handle to stream. It is kept for callers
// that prepare a capture explicitly; ordinary calls follow the thread's current
// stream already.
func SetBLASStream(stream *cuda.Stream) error {
	handle := blas()
	if err := handle.SetStream(stream); err != nil {
		return err
	}
	if value, ok := blasHandles.Load(cuda.ThreadID()); ok {
		value.(*threadBLAS).stream = stream
	}
	return nil
}
func matrixBatchStride(shape, batch []int, matrixSize int) (int64, bool) {
	if numel(shape) == 1 {
		return 0, true
	}
	if len(shape) != len(batch) {
		return 0, false
	}
	for i, d := range shape {
		if d != batch[i] {
			return 0, false
		}
	}
	return int64(matrixSize), true
}
func gpuMatMul(a, b *Tensor, s []int, m, n, k int, batch []int) *Tensor {
	options := executionOptions(a, b)
	out := mustAlloc(numel(s))
	ab := a.Shape[:len(a.Shape)-2]
	bb := b.Shape[:len(b.Shape)-2]
	count := numel(batch)
	as, aRegular := matrixBatchStride(ab, batch, m*k)
	bs, bRegular := matrixBatchStride(bb, batch, k*n)
	regular := aRegular && bRegular && count > 1
	var ap16, bp16 uintptr
	var aLease, bLease *tensorStorage
	if options.BF16Autocast && regular {
		ap16, aLease = a.ensureBF16Lease()
		bp16, bLease = b.ensureBF16Lease()
	}
	if regular {
		timedGPU("matmul_strided", func() error {
			return options.gemmBatchedPrepared(m, n, k, false, false, a.buf.Pointer(), ap16, as, b.buf.Pointer(), bp16, bs, out.Pointer(), int64(m*n), count, 0)
		})
	} else {
		for z := 0; z < count; z++ {
			ai := bindex(z, batch, ab) * m * k
			bi := bindex(z, batch, bb) * k * n
			timedGPU("matmul_sgemm", func() error {
				return options.gemmSingle(m, n, k, false, false, a.buf.Pointer()+uintptr(ai*4), b.buf.Pointer()+uintptr(bi*4), out.Pointer()+uintptr(z*m*n*4), 0)
			})
		}
	}
	r := resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		da, db := a.ensureGradGPU(), b.ensureGradGPU()
		var g16 *cuda.Buffer
		if options.BF16Autocast && regular && (da != nil || db != nil) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		gp16 := ptr(g16)
		if regular && da != nil && as != 0 {
			timedGPU("matmul_dx_strided", func() error {
				return options.gemmBatchedPrepared(m, k, n, false, true, g.Pointer(), gp16, int64(m*n), b.buf.Pointer(), bp16, bs, da.Pointer(), as, count, 1)
			})
		}
		if regular && db != nil {
			if bs == 0 && as != 0 {
				timedGPU("matmul_dw_flat", func() error {
					return options.gemmSinglePrepared(k, n, m*count, true, false, a.buf.Pointer(), ap16, g.Pointer(), gp16, db.Pointer(), 1)
				})
			} else if bs != 0 {
				timedGPU("matmul_dw_strided", func() error {
					return options.gemmBatchedPrepared(k, n, m, true, false, a.buf.Pointer(), ap16, as, g.Pointer(), gp16, int64(m*n), db.Pointer(), bs, count, 1)
				})
			}
		}
		if regular && (da == nil || as != 0) && (db == nil || bs != 0 || as != 0) {
			return
		}
		for z := 0; z < count; z++ {
			ai := bindex(z, batch, ab) * m * k
			bi := bindex(z, batch, bb) * k * n
			gp := g.Pointer() + uintptr(z*m*n*4)
			if da != nil && (!regular || as == 0) {
				timedGPU("matmul_dx_sgemm", func() error {
					return options.gemmSingle(m, k, n, false, true, gp, b.buf.Pointer()+uintptr(bi*4), da.Pointer()+uintptr(ai*4), 1)
				})
			}
			if db != nil && (!regular || (bs == 0 && as == 0)) {
				timedGPU("matmul_dw_sgemm", func() error {
					return options.gemmSingle(k, n, m, true, false, a.buf.Pointer()+uintptr(ai*4), gp, db.Pointer()+uintptr(bi*4), 1)
				})
			}
		}
	})
	if aLease != nil {
		r.savedLeases = append(r.savedLeases, aLease)
	}
	if bLease != nil {
		r.savedLeases = append(r.savedLeases, bLease)
	}
	return r
}
func gpuConv1d(x, w, b *Tensor, dilation int) *Tensor {
	options := executionOptions(x, w, b)
	s := x.Shape
	bs, t, ci, co, k := s[0], s[1], s[2], w.Shape[0], w.Shape[2]
	rows, depth := bs*t, ci*k
	cols := mustAlloc(rows * depth)
	out := mustAlloc(rows * co)
	timedGPU("conv_im2col", func() error { return kernels.Im2col(x.buf, cols, bs, ci, t, k, dilation) })
	var cols16 *cuda.Buffer
	var wp16 uintptr
	var wLease *tensorStorage
	if options.BF16Autocast {
		cols16 = castBF16(cols.Pointer(), rows*depth)
		wp16, wLease = w.ensureBF16Lease()
	}
	timedGPU("conv_forward_sgemm", func() error {
		return options.gemmSinglePrepared(rows, co, depth, false, true, cols.Pointer(), ptr(cols16), w.buf.Pointer(), wp16, out.Pointer(), 0)
	})
	timedGPU("conv_bias", func() error { return kernels.AddBiasColumns(out, b.buf, rows, co) })
	r := resultGPU(out, []int{bs, t, co}, []*Tensor{x, w, b}, func(g *cuda.Buffer) {
		var g16 *cuda.Buffer
		if options.BF16Autocast && (w.RequiresGrad || x.RequiresGrad) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		if dw := w.ensureGradGPU(); dw != nil {
			timedGPU("conv_dw_sgemm", func() error {
				return options.gemmSinglePrepared(co, depth, rows, true, false, g.Pointer(), ptr(g16), cols.Pointer(), ptr(cols16), dw.Pointer(), 1)
			})
		}
		if db := b.ensureGradGPU(); db != nil {
			timedGPU("conv_db", func() error { return kernels.ColumnSum(g, db, rows, co) })
		}
		if dx := x.ensureGradGPU(); dx != nil {
			dcols := mustAlloc(rows * depth)
			defer dcols.Free()
			timedGPU("conv_dx_sgemm", func() error {
				return options.gemmSinglePrepared(rows, depth, co, false, false, g.Pointer(), ptr(g16), w.buf.Pointer(), wp16, dcols.Pointer(), 0)
			})
			timedGPU("conv_col2im", func() error { return kernels.Col2im(dcols, dx, bs, ci, t, k, dilation) })
		}
	})
	r.aux = []*cuda.Buffer{cols}
	if cols16 != nil {
		r.aux = append(r.aux, cols16)
	}
	if wLease != nil {
		r.savedLeases = append(r.savedLeases, wLease)
	}
	return r
}
