package autograd

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// BiasGELU combines the feed-forward bias and exact GELU on CUDA.
func BiasGELU(x, b *Tensor) *Tensor {
	if x.Device != tensor.CUDA {
		return GELU(Add(x, b), false)
	}
	same(x, b)
	x, b = x.Contiguous(), b.Contiguous()
	c := x.Shape[len(x.Shape)-1]
	if len(b.Shape) != 1 || b.Numel() != c {
		panic("autograd: bias GELU shape")
	}
	n := int32(x.Numel())
	channels := int32(c)
	out := mustAlloc(int(n))
	xp, bp, yp := ptr(x.buf), ptr(b.buf), ptr(out)
	launch("bias_gelu_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	return resultGPU(out, x.Shape, []*Tensor{x, b}, func(g *cuda.Buffer) {
		gp := ptr(g)
		dx, db := ptr(x.ensureGradGPU()), ptr(b.ensureGradGPU())
		launch("bias_gelu_b", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&db), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	})
}

// BiasResidual combines a channel bias and a residual addition on CUDA.
func BiasResidual(x, b, residual *Tensor) *Tensor {
	if x.Device != tensor.CUDA {
		return Add(Add(x, b), residual)
	}
	same(x, b)
	same(x, residual)
	x, b, residual = x.Contiguous(), b.Contiguous(), residual.Contiguous()
	c := x.Shape[len(x.Shape)-1]
	if len(b.Shape) != 1 || b.Numel() != c || len(x.Shape) != len(residual.Shape) {
		panic("autograd: bias residual shape")
	}
	for i, d := range x.Shape {
		if residual.Shape[i] != d {
			panic("autograd: bias residual shape")
		}
	}
	n, channels := int32(x.Numel()), int32(c)
	out := mustAlloc(int(n))
	xp, bp, rp, yp := ptr(x.buf), ptr(b.buf), ptr(residual.buf), ptr(out)
	launch("bias_residual_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&rp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	return resultGPU(out, x.Shape, []*Tensor{x, b, residual}, func(g *cuda.Buffer) {
		gp := ptr(g)
		dx, db, dr := ptr(x.ensureGradGPU()), ptr(b.ensureGradGPU()), ptr(residual.ensureGradGPU())
		launch("bias_residual_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&db), unsafe.Pointer(&dr), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	})
}
