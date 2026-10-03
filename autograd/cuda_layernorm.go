package autograd

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

func gpuLayerNorm(x, w, b *Tensor, eps float32) *Tensor {
	rows := x.Numel() / w.Numel()
	dim := w.Numel()
	stats, out := mustAlloc(rows*2), mustAlloc(x.Numel())
	xp, wp, bp, yp, sp := ptr(x.buf), ptr(w.buf), ptr(b.buf), ptr(out), ptr(stats)
	r, d := int32(rows), int32(dim)
	launch("layernorm_f", rows*32, unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sp), unsafe.Pointer(&r), unsafe.Pointer(&d), unsafe.Pointer(&eps))
	res := resultGPU(out, x.Shape, []*Tensor{x, w, b}, func(g *cuda.Buffer) {
		gp := ptr(g)
		dx, dw, db := ptr(x.ensureGradGPU()), ptr(w.ensureGradGPU()), ptr(b.ensureGradGPU())
		launch("layernorm_b", rows*32, unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&gp), unsafe.Pointer(&sp), unsafe.Pointer(&dx), unsafe.Pointer(&dw), unsafe.Pointer(&db), unsafe.Pointer(&r), unsafe.Pointer(&d))
	})
	res.aux = []*cuda.Buffer{stats}
	return res
}
