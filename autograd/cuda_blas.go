package autograd

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
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
func gpuMatMul(a, b *Tensor, s []int, m, n, k int, batch []int, transB bool) *Tensor {
	options := executionOptions(a, b)
	out := mustAlloc(numel(s))
	ab := a.Shape[:len(a.Shape)-2]
	bb := b.Shape[:len(b.Shape)-2]
	count := numel(batch)
	as, aRegular := matrixBatchStride(ab, batch, m*k)
	bs, bRegular := matrixBatchStride(bb, batch, k*n)
	regular := aRegular && bRegular && count > 1
	flatLinear := regular && as != 0 && bs == 0
	prepared := regular || count == 1
	nonEmpty := count > 0 && m > 0 && n > 0 && k > 0
	var ap16, bp16 uintptr
	var aLease, bLease *tensorStorage
	if options.BF16Autocast && prepared && nonEmpty {
		ap16, aLease = a.ensureBF16Lease()
		bp16, bLease = b.ensureBF16Lease()
	}
	if count > 0 && m > 0 && n > 0 {
		if k == 0 {
			if err := out.Memset(0, out.Size()); err != nil {
				panic(err)
			}
		} else if flatLinear || count == 1 {
			label := "matmul_sgemm"
			if flatLinear {
				label = "matmul_strided"
			}
			timedGPU(label, func() error {
				return options.gemmSinglePrepared(count*m, n, k, false, transB, a.buf.Pointer(), ap16, b.buf.Pointer(), bp16, out.Pointer(), 0)
			})
		} else if regular {
			timedGPU("matmul_strided", func() error {
				return options.gemmBatchedPrepared(m, n, k, false, transB, a.buf.Pointer(), ap16, as, b.buf.Pointer(), bp16, bs, out.Pointer(), int64(m*n), count, 0)
			})
		} else {
			for z := 0; z < count; z++ {
				ai := bindex(z, batch, ab) * m * k
				bi := bindex(z, batch, bb) * k * n
				timedGPU("matmul_sgemm", func() error {
					return options.gemmSingle(m, n, k, false, transB, a.buf.Pointer()+uintptr(ai*4), b.buf.Pointer()+uintptr(bi*4), out.Pointer()+uintptr(z*m*n*4), 0)
				})
			}
		}
	}
	r := resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		if !nonEmpty {
			a.ensureGradGPU()
			b.ensureGradGPU()
			return
		}
		var g16 *cuda.Buffer
		if options.BF16Autocast && prepared && (a.RequiresGrad || b.RequiresGrad) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		gp16 := ptr(g16)
		if flatLinear || count == 1 {
			if a.RequiresGrad {
				da, betaA := a.allocGradGPU()
				timedGPU("matmul_dx_flat", func() error {
					return options.gemmSinglePrepared(count*m, k, n, false, !transB, g.Pointer(), gp16, b.buf.Pointer(), bp16, da.Pointer(), betaA)
				})
			}
			if b.RequiresGrad {
				db, betaB := b.allocGradGPU()
				timedGPU("matmul_dw_flat", func() error {
					return options.gemmSinglePrepared(k, n, count*m, true, false, a.buf.Pointer(), ap16, g.Pointer(), gp16, db.Pointer(), betaB)
				})
			}
			return
		}
		if regular && a.RequiresGrad && as != 0 {
			da, betaA := a.allocGradGPU()
			timedGPU("matmul_dx_strided", func() error {
				return options.gemmBatchedPrepared(m, k, n, false, !transB, g.Pointer(), gp16, int64(m*n), b.buf.Pointer(), bp16, bs, da.Pointer(), as, count, betaA)
			})
		}
		if regular && b.RequiresGrad && bs != 0 {
			db, betaB := b.allocGradGPU()
			timedGPU("matmul_dw_strided", func() error {
				return options.gemmBatchedPrepared(k, n, m, true, false, a.buf.Pointer(), ap16, as, g.Pointer(), gp16, int64(m*n), db.Pointer(), bs, count, betaB)
			})
		}
		needA := a.RequiresGrad && (!regular || as == 0)
		needB := b.RequiresGrad && (!regular || bs == 0)
		if !needA && !needB {
			return
		}
		var da, db *cuda.Buffer
		if needA {
			da = a.ensureGradGPU()
		}
		if needB {
			db = b.ensureGradGPU()
		}
		for z := 0; z < count; z++ {
			ai := bindex(z, batch, ab) * m * k
			bi := bindex(z, batch, bb) * k * n
			gp := g.Pointer() + uintptr(z*m*n*4)
			if needA {
				timedGPU("matmul_dx_sgemm", func() error {
					return options.gemmSingle(m, k, n, false, !transB, gp, b.buf.Pointer()+uintptr(bi*4), da.Pointer()+uintptr(ai*4), 1)
				})
			}
			if needB {
				timedGPU("matmul_dw_sgemm", func() error {
					return options.gemmSingle(k, n, m, true, false, a.buf.Pointer()+uintptr(ai*4), gp, db.Pointer()+uintptr(bi*4), 1)
				})
			}
		}
	})
	if r.RequiresGrad {
		if aLease != nil {
			r.savedLeases = append(r.savedLeases, aLease)
		}
		if bLease != nil {
			r.savedLeases = append(r.savedLeases, bLease)
		}
	} else {
		if aLease != nil {
			aLease.release()
		}
		if bLease != nil {
			bLease.release()
		}
	}
	return r
}

func gpuMatMulTransB(a, w *Tensor, s []int, rows, n, k int) *Tensor {
	options := executionOptions(a, w)
	out := mustAlloc(rows * n)
	nonEmpty := rows > 0 && n > 0 && k > 0
	var ap16, wp16 uintptr
	var aLease, wLease *tensorStorage
	if options.BF16Autocast && nonEmpty {
		ap16, aLease = a.ensureBF16Lease()
		wp16, wLease = w.ensureBF16Lease()
	}
	if rows > 0 && n > 0 {
		if k == 0 {
			if err := out.Memset(0, out.Size()); err != nil {
				panic(err)
			}
		} else {
			label := "matmul_sgemm"
			if len(s) > 2 {
				label = "matmul_strided"
			}
			timedGPU(label, func() error {
				return options.gemmSinglePrepared(rows, n, k, false, true, a.buf.Pointer(), ap16, w.buf.Pointer(), wp16, out.Pointer(), 0)
			})
		}
	}
	r := resultGPU(out, s, []*Tensor{a, w}, func(g *cuda.Buffer) {
		if !nonEmpty {
			a.ensureGradGPU()
			w.ensureGradGPU()
			return
		}
		var g16 *cuda.Buffer
		if options.BF16Autocast && (a.RequiresGrad || w.RequiresGrad) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		gp16 := ptr(g16)
		if a.RequiresGrad {
			da, betaA := a.allocGradGPU()
			timedGPU("matmul_dx_linear", func() error {
				return options.gemmSinglePrepared(rows, k, n, false, false, g.Pointer(), gp16, w.buf.Pointer(), wp16, da.Pointer(), betaA)
			})
		}
		if w.RequiresGrad {
			dw, betaW := w.allocGradGPU()
			timedGPU("matmul_dw_linear", func() error {
				return options.gemmSinglePrepared(n, k, rows, true, false, g.Pointer(), gp16, a.buf.Pointer(), ap16, dw.Pointer(), betaW)
			})
		}
	})
	if r.RequiresGrad {
		if aLease != nil {
			r.savedLeases = append(r.savedLeases, aLease)
		}
		if wLease != nil {
			r.savedLeases = append(r.savedLeases, wLease)
		}
	} else {
		if aLease != nil {
			aLease.release()
		}
		if wLease != nil {
			wLease.release()
		}
	}
	return r
}

func gpuSplitHeads(x *Tensor, b, t, heads, dim, parts, which int) *Tensor {
	x = x.Contiguous()
	options := executionOptions(x)
	n := b * heads * t * dim
	out := mustAlloc(n)
	var shadow *cuda.Buffer
	if options.BF16Autocast {
		shadow = allocBF16(n)
	}
	xp, yp := ptr(x.buf), ptr(out)
	n32, t32, h32, d32, p32, w32 := int32(n), int32(t), int32(heads), int32(dim), int32(parts), int32(which)
	if options.BF16Autocast {
		sh := ptr(shadow)
		launch("split_heads_bf16_f", n, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&d32), unsafe.Pointer(&p32), unsafe.Pointer(&w32))
	} else {
		launch("split_heads_f", n, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&n32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&d32), unsafe.Pointer(&p32), unsafe.Pointer(&w32))
	}
	res := resultGPU(out, []int{b, heads, t, dim}, []*Tensor{x}, func(g *cuda.Buffer) {
		var dx *cuda.Buffer
		add := int32(1)
		if parts == 1 {
			var beta float32
			dx, beta = x.allocGradGPU()
			if beta == 0 {
				add = 0
			}
		} else {
			dx = x.ensureGradGPU()
		}
		if dx == nil {
			return
		}
		gp, dp := ptr(g), ptr(dx)
		launch("split_heads_b", n, unsafe.Pointer(&gp), unsafe.Pointer(&dp), unsafe.Pointer(&n32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&d32), unsafe.Pointer(&p32), unsafe.Pointer(&w32), unsafe.Pointer(&add))
	})
	res.setBF16(shadow)
	return res
}

func gpuMergeHeads(x *Tensor, b, t, heads, dim int) *Tensor {
	x = x.Contiguous()
	options := executionOptions(x)
	n := b * t * heads * dim
	out := mustAlloc(n)
	var shadow *cuda.Buffer
	if options.BF16Autocast {
		shadow = allocBF16(n)
	}
	xp, yp := ptr(x.buf), ptr(out)
	n32, t32, h32, d32 := int32(n), int32(t), int32(heads), int32(dim)
	if options.BF16Autocast {
		sh := ptr(shadow)
		launch("merge_heads_bf16_f", n, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&d32))
	} else {
		launch("merge_heads_f", n, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&n32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&d32))
	}
	res := resultGPU(out, []int{b, t, heads * dim}, []*Tensor{x}, func(g *cuda.Buffer) {
		dx, beta := x.allocGradGPU()
		if dx == nil {
			return
		}
		add := int32(0)
		if beta != 0 {
			add = 1
		}
		gp, dp := ptr(g), ptr(dx)
		launch("merge_heads_b", n, unsafe.Pointer(&gp), unsafe.Pointer(&dp), unsafe.Pointer(&n32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&d32), unsafe.Pointer(&add))
	})
	res.setBF16(shadow)
	return res
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
	if b != nil {
		timedGPU("conv_bias", func() error { return kernels.AddBiasColumns(out, b.buf, rows, co) })
	}
	parents := []*Tensor{x, w}
	if b != nil {
		parents = append(parents, b)
	}
	r := resultGPU(out, []int{bs, t, co}, parents, func(g *cuda.Buffer) {
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
		if b != nil {
			if db := b.ensureGradGPU(); db != nil {
				timedGPU("conv_db", func() error { return kernels.ColumnSum(g, db, rows, co) })
			}
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
	if r.RequiresGrad {
		r.aux = []*cuda.Buffer{cols}
		if cols16 != nil {
			r.aux = append(r.aux, cols16)
		}
		if wLease != nil {
			r.savedLeases = append(r.savedLeases, wLease)
		}
	} else {
		cols.Free()
		if cols16 != nil {
			cols16.Free()
		}
		if wLease != nil {
			wLease.release()
		}
	}
	return r
}
