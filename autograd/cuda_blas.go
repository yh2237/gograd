package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"sync"
)

var blasOnce sync.Once
var blasHandle *cuda.Blas
var blasErr error

func blas() *cuda.Blas {
	blasOnce.Do(func() { blasHandle, blasErr = cuda.NewBlas() })
	if blasErr != nil {
		panic(blasErr)
	}
	return blasHandle
}

// SetBLASStream routes autograd's cuBLAS handle during CUDA graph capture.
// Passing nil restores the default stream.
func SetBLASStream(stream *cuda.Stream) error { return blas().SetStream(stream) }
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
	out := mustAlloc(numel(s))
	ab := a.Shape[:len(a.Shape)-2]
	bb := b.Shape[:len(b.Shape)-2]
	count := numel(batch)
	as, aRegular := matrixBatchStride(ab, batch, m*k)
	bs, bRegular := matrixBatchStride(bb, batch, k*n)
	regular := aRegular && bRegular && count > 1
	var ap16, bp16 uintptr
	if BF16Autocast && regular {
		ap16 = ptr(a.ensureBF16())
		bp16 = ptr(b.ensureBF16())
	}
	if regular {
		timedGPU("matmul_strided", func() error {
			return gemmBatchedPrepared(m, n, k, false, false, a.buf.Pointer(), ap16, as, b.buf.Pointer(), bp16, bs, out.Pointer(), int64(m*n), count, 0)
		})
	} else {
		for z := 0; z < count; z++ {
			ai := bindex(z, batch, ab) * m * k
			bi := bindex(z, batch, bb) * k * n
			timedGPU("matmul_sgemm", func() error {
				return gemmSingle(m, n, k, false, false, a.buf.Pointer()+uintptr(ai*4), b.buf.Pointer()+uintptr(bi*4), out.Pointer()+uintptr(z*m*n*4), 0)
			})
		}
	}
	return resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		da, db := a.ensureGradGPU(), b.ensureGradGPU()
		var g16 *cuda.Buffer
		if BF16Autocast && regular && (da != nil || db != nil) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		gp16 := ptr(g16)
		if regular && da != nil && as != 0 {
			timedGPU("matmul_dx_strided", func() error {
				return gemmBatchedPrepared(m, k, n, false, true, g.Pointer(), gp16, int64(m*n), b.buf.Pointer(), bp16, bs, da.Pointer(), as, count, 1)
			})
		}
		if regular && db != nil {
			if bs == 0 && as != 0 {
				timedGPU("matmul_dw_flat", func() error {
					return gemmSinglePrepared(k, n, m*count, true, false, a.buf.Pointer(), ap16, g.Pointer(), gp16, db.Pointer(), 1)
				})
			} else if bs != 0 {
				timedGPU("matmul_dw_strided", func() error {
					return gemmBatchedPrepared(k, n, m, true, false, a.buf.Pointer(), ap16, as, g.Pointer(), gp16, int64(m*n), db.Pointer(), bs, count, 1)
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
					return gemmSingle(m, k, n, false, true, gp, b.buf.Pointer()+uintptr(bi*4), da.Pointer()+uintptr(ai*4), 1)
				})
			}
			if db != nil && (!regular || (bs == 0 && as == 0)) {
				timedGPU("matmul_dw_sgemm", func() error {
					return gemmSingle(k, n, m, true, false, a.buf.Pointer()+uintptr(ai*4), gp, db.Pointer()+uintptr(bi*4), 1)
				})
			}
		}
	})
}
func gpuConv1d(x, w, b *Tensor, dilation int) *Tensor {
	s := x.Shape
	bs, t, ci, co, k := s[0], s[1], s[2], w.Shape[0], w.Shape[2]
	rows, depth := bs*t, ci*k
	cols := mustAlloc(rows * depth)
	out := mustAlloc(rows * co)
	timedGPU("conv_im2col", func() error { return kernels.Im2col(x.buf, cols, bs, ci, t, k, dilation) })
	var cols16 *cuda.Buffer
	var wp16 uintptr
	if BF16Autocast {
		cols16 = castBF16(cols.Pointer(), rows*depth)
		wp16 = ptr(w.ensureBF16())
	}
	timedGPU("conv_forward_sgemm", func() error {
		return gemmSinglePrepared(rows, co, depth, false, true, cols.Pointer(), ptr(cols16), w.buf.Pointer(), wp16, out.Pointer(), 0)
	})
	timedGPU("conv_bias", func() error { return kernels.AddBiasColumns(out, b.buf, rows, co) })
	r := resultGPU(out, []int{bs, t, co}, []*Tensor{x, w, b}, func(g *cuda.Buffer) {
		defer cols.Free()
		var g16 *cuda.Buffer
		if BF16Autocast && (w.RequiresGrad || x.RequiresGrad) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		if dw := w.ensureGradGPU(); dw != nil {
			timedGPU("conv_dw_sgemm", func() error {
				return gemmSinglePrepared(co, depth, rows, true, false, g.Pointer(), ptr(g16), cols.Pointer(), ptr(cols16), dw.Pointer(), 1)
			})
		}
		if db := b.ensureGradGPU(); db != nil {
			timedGPU("conv_db", func() error { return kernels.ColumnSum(g, db, rows, co) })
		}
		if dx := x.ensureGradGPU(); dx != nil {
			dcols := mustAlloc(rows * depth)
			defer dcols.Free()
			timedGPU("conv_dx_sgemm", func() error {
				return gemmSinglePrepared(rows, depth, co, false, false, g.Pointer(), ptr(g16), w.buf.Pointer(), wp16, dcols.Pointer(), 0)
			})
			timedGPU("conv_col2im", func() error { return kernels.Col2im(dcols, dx, bs, ci, t, k, dilation) })
		}
	})
	r.aux = []*cuda.Buffer{cols}
	if cols16 != nil {
		r.aux = append(r.aux, cols16)
	}
	return r
}
