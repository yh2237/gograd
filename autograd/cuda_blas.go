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
func gpuMatMul(a, b *Tensor, s []int, m, n, k int, batch []int) *Tensor {
	out := mustAlloc(numel(s))
	ab := a.Shape[:len(a.Shape)-2]
	bb := b.Shape[:len(b.Shape)-2]
	for z := 0; z < numel(batch); z++ {
		ai := bindex(z, batch, ab) * m * k
		bi := bindex(z, batch, bb) * k * n
		timedGPU("matmul_sgemm", func() error {
			return blas().SgemmRowMajor(m, n, k, 1, a.buf.Pointer()+uintptr(ai*4), k, b.buf.Pointer()+uintptr(bi*4), n, 0, out.Pointer()+uintptr(z*m*n*4), n)
		})
	}
	return resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		da, db := a.ensureGradGPU(), b.ensureGradGPU()
		for z := 0; z < numel(batch); z++ {
			ai := bindex(z, batch, ab) * m * k
			bi := bindex(z, batch, bb) * k * n
			gp := g.Pointer() + uintptr(z*m*n*4)
			if da != nil {
				timedGPU("matmul_dx_sgemm", func() error {
					return blas().SgemmRowMajorNT(m, k, n, 1, gp, n, b.buf.Pointer()+uintptr(bi*4), n, 1, da.Pointer()+uintptr(ai*4), k)
				})
			}
			if db != nil {
				timedGPU("matmul_dw_sgemm", func() error {
					return blas().SgemmRowMajorTransposeA(k, n, m, 1, a.buf.Pointer()+uintptr(ai*4), k, gp, n, 1, db.Pointer()+uintptr(bi*4), n)
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
	timedGPU("conv_forward_sgemm", func() error {
		return blas().SgemmRowMajorNT(rows, co, depth, 1, cols.Pointer(), depth, w.buf.Pointer(), depth, 0, out.Pointer(), co)
	})
	timedGPU("conv_bias", func() error { return kernels.AddBiasColumns(out, b.buf, rows, co) })
	r := resultGPU(out, []int{bs, t, co}, []*Tensor{x, w, b}, func(g *cuda.Buffer) {
		defer cols.Free()
		if dw := w.ensureGradGPU(); dw != nil {
			timedGPU("conv_dw_sgemm", func() error {
				return blas().SgemmRowMajorTransposeA(co, depth, rows, 1, g.Pointer(), co, cols.Pointer(), depth, 1, dw.Pointer(), depth)
			})
		}
		if db := b.ensureGradGPU(); db != nil {
			timedGPU("conv_db", func() error { return kernels.ColumnSum(g, db, rows, co) })
		}
		if dx := x.ensureGradGPU(); dx != nil {
			dcols := mustAlloc(rows * depth)
			defer dcols.Free()
			timedGPU("conv_dx_sgemm", func() error {
				return blas().SgemmRowMajor(rows, depth, co, 1, g.Pointer(), co, w.buf.Pointer(), depth, 0, dcols.Pointer(), depth)
			})
			timedGPU("conv_col2im", func() error { return kernels.Col2im(dcols, dx, bs, ci, t, k, dilation) })
		}
	})
	r.aux = []*cuda.Buffer{cols}
	return r
}
