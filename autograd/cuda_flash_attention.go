package autograd

import (
	"math"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

func launchFlash(name string, blocks int, args ...unsafe.Pointer) {
	timedGPU(name, func() error {
		return graphKernel(name).Launch([3]int{blocks, 1, 1}, [3]int{256, 1, 1}, 0, nil, args)
	})
}

// FlashAttention keeps only output and per-query online-softmax statistics.
// Backward recomputes scores and probabilities instead of retaining QK^T.
func gpuFlashAttention(q, k, v, mask *Tensor) *Tensor {
	q, k, v = q.Contiguous(), k.Contiguous(), v.Contiguous()
	if mask != nil {
		same(q, mask)
		mask = mask.Contiguous()
	}
	rank := len(q.Shape)
	batch := q.Shape[:rank-2]
	nq, d := q.Shape[rank-2], q.Shape[rank-1]
	nk, e := k.Shape[rank-2], v.Shape[rank-1]
	if d > 128 || e > 128 {
		panic("autograd: tiled attention head dimension exceeds 128")
	}
	rows := numel(batch) * nq
	scoreShape := append(append([]int(nil), batch...), nq, nk)
	outShape := append(append([]int(nil), batch...), nq, e)
	shape, ms := viewMeta(scoreShape, make([]int, len(scoreShape)))
	var maskPtr uintptr
	if mask != nil {
		maskPtr = ptr(mask.buf)
		mapping := make([]int, len(scoreShape))
		pad := len(scoreShape) - len(mask.Shape)
		for i := range scoreShape {
			if i >= pad && mask.Shape[i-pad] != 1 {
				mapping[i] = mask.Strides[i-pad]
			}
		}
		_, ms = viewMeta(scoreShape, mapping)
	}
	out, stats := mustAlloc(rows*e), mustAlloc(rows*2)
	qa, ka, va, oa, sa := ptr(q.buf), ptr(k.buf), ptr(v.buf), ptr(out), ptr(stats)
	n, m, depth, width := int32(nq), int32(nk), int32(d), int32(e)
	qblocks := int32((nq + 7) / 8)
	blocks := numel(batch) * int(qblocks)
	scale := float32(1 / math.Sqrt(float64(d)))
	launchFlash("flash_attention_tiled_f", blocks, unsafe.Pointer(&qa), unsafe.Pointer(&ka), unsafe.Pointer(&va), unsafe.Pointer(&maskPtr), unsafe.Pointer(&oa), unsafe.Pointer(&sa), unsafe.Pointer(&n), unsafe.Pointer(&m), unsafe.Pointer(&depth), unsafe.Pointer(&width), unsafe.Pointer(&qblocks), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
	parents := []*Tensor{q, k, v}
	if mask != nil {
		parents = append(parents, mask)
	}
	result := resultGPU(out, outShape, parents, func(g *cuda.Buffer) {
		if !q.RequiresGrad && !k.RequiresGrad && !v.RequiresGrad && (mask == nil || !mask.RequiresGrad) {
			return
		}
		gp := ptr(g)
		dq, dk, dv := ptr(q.ensureGradGPU()), ptr(k.ensureGradGPU()), ptr(v.ensureGradGPU())
		var dm uintptr
		if mask != nil {
			dm = ptr(mask.ensureGradGPU())
		}
		launchFlash("flash_attention_tiled_b", blocks, unsafe.Pointer(&qa), unsafe.Pointer(&ka), unsafe.Pointer(&va), unsafe.Pointer(&maskPtr), unsafe.Pointer(&oa), unsafe.Pointer(&sa), unsafe.Pointer(&gp), unsafe.Pointer(&dq), unsafe.Pointer(&dk), unsafe.Pointer(&dv), unsafe.Pointer(&dm), unsafe.Pointer(&n), unsafe.Pointer(&m), unsafe.Pointer(&depth), unsafe.Pointer(&width), unsafe.Pointer(&qblocks), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
	})
	result.aux = []*cuda.Buffer{stats}
	return result
}
