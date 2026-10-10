package autograd

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// BiasGELU combines the feed-forward bias and exact GELU on CUDA.
func BiasGELU(x, b *Tensor) *Tensor {
	if !dispatchBackend("bias_gelu", x.Device) {
		return GELU(Add(x, b), false)
	}
	same(x, b)
	options := executionOptions(x, b)
	x, b = x.Contiguous(), b.Contiguous()
	c := x.Shape[len(x.Shape)-1]
	if len(b.Shape) != 1 || b.Numel() != c {
		panic("autograd: bias GELU shape")
	}
	n := int32(x.Numel())
	channels := int32(c)
	out := mustAlloc(int(n))
	var shadow *cuda.Buffer
	if options.BF16Autocast {
		shadow = allocBF16(int(n))
	}
	xp, bp, yp := ptr(x.buf), ptr(b.buf), ptr(out)
	if options.BF16Autocast {
		sh := ptr(shadow)
		launch("bias_gelu_bf16_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	} else {
		launch("bias_gelu_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	}
	res := resultGPU(out, x.Shape, []*Tensor{x, b}, func(g *cuda.Buffer) {
		gp := ptr(g)
		dx, db := ptr(x.ensureGradGPU()), ptr(b.ensureGradGPU())
		launch("bias_gelu_b", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&db), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	})
	res.setBF16(shadow)
	return res
}

// BiasResidual combines a channel bias and a residual addition on CUDA.
func BiasResidual(x, b, residual *Tensor) *Tensor {
	if !dispatchBackend("bias_residual", x.Device) {
		return Add(Add(x, b), residual)
	}
	same(x, b, residual)
	options := executionOptions(x, b, residual)
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
	var shadow *cuda.Buffer
	if options.BF16Autocast {
		shadow = allocBF16(int(n))
	}
	xp, bp, rp, yp := ptr(x.buf), ptr(b.buf), ptr(residual.buf), ptr(out)
	if options.BF16Autocast {
		sh := ptr(shadow)
		launch("bias_residual_bf16_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&rp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	} else {
		launch("bias_residual_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&rp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	}
	res := resultGPU(out, x.Shape, []*Tensor{x, b, residual}, func(g *cuda.Buffer) {
		gp := ptr(g)
		dx, db, dr := ptr(x.ensureGradGPU()), ptr(b.ensureGradGPU()), ptr(residual.ensureGradGPU())
		launch("bias_residual_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&db), unsafe.Pointer(&dr), unsafe.Pointer(&n), unsafe.Pointer(&channels))
	})
	res.setBF16(shadow)
	return res
}
