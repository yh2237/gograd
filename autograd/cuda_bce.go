package autograd

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// gpuBCELogits is an elementwise node; weights and reductions remain ordinary
// differentiable graph operations. Empty tensors do not launch a kernel. The
// backward kernel adds into each non-NULL destination (including aliased ones).
func gpuBCELogits(x, target *Tensor) *Tensor {
	n := int32(x.Numel())
	out := mustAlloc(int(n))
	xp, tp, yp := ptr(x.buf), ptr(target.buf), ptr(out)
	if n > 0 {
		launch("bce_logits_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&tp), unsafe.Pointer(&yp), unsafe.Pointer(&n))
	}
	return resultGPU(out, x.Shape, []*Tensor{x, target}, func(g *cuda.Buffer) {
		if n == 0 {
			return
		}
		dx, dt := ptr(x.ensureGradGPU()), ptr(target.ensureGradGPU())
		if dx == 0 && dt == 0 {
			return
		}
		gp := ptr(g)
		launch("bce_logits_b", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&tp), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&dt), unsafe.Pointer(&n))
	})
}
