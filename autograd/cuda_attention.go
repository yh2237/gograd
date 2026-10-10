package autograd

import (
	"math"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// gpuAttention keeps QK, the masked softmax, PV, and their adjoints in one
// autograd node. Matrix products use one strided cuBLAS call per phase.
func gpuAttention(q, k, v, mask *Tensor, options ExecutionOptions) *Tensor {
	q, k, v = q.Contiguous(), k.Contiguous(), v.Contiguous()
	if mask != nil {
		same(q, mask)
		mask = mask.Contiguous()
	}
	rank := len(q.Shape)
	batch := q.Shape[:rank-2]
	m, d := q.Shape[rank-2], q.Shape[rank-1]
	n, e := k.Shape[rank-2], v.Shape[rank-1]
	count := numel(batch)
	scoreShape := append(append([]int(nil), batch...), m, n)
	outShape := append(append([]int(nil), batch...), m, e)
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
	scores, probs, out := mustAlloc(count*m*n), mustAlloc(count*m*n), mustAlloc(count*m*e)
	qa, ka, va, sa, pa, oa := ptr(q.buf), ptr(k.buf), ptr(v.buf), ptr(scores), ptr(probs), ptr(out)
	var q16, k16, v16 uintptr
	if options.BF16Autocast {
		q16 = ptr(q.ensureBF16())
		k16 = ptr(k.ensureBF16())
		v16 = ptr(v.ensureBF16())
	}
	timedGPU("attn_qk", func() error {
		return options.gemmBatchedPrepared(m, n, d, false, true, qa, q16, int64(m*d), ka, k16, int64(n*d), sa, int64(m*n), count, 0)
	})
	rows, dim, scale := int32(count*m), int32(n), float32(1/math.Sqrt(float64(d)))
	launch("attn_softmax_f", int(rows)*32, unsafe.Pointer(&sa), unsafe.Pointer(&maskPtr), unsafe.Pointer(&pa), unsafe.Pointer(&rows), unsafe.Pointer(&dim), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
	scores.Free()
	var probs16 *cuda.Buffer
	if options.BF16Autocast {
		probs16 = castBF16(pa, count*m*n)
	}
	timedGPU("attn_pv", func() error {
		return options.gemmBatchedPrepared(m, e, n, false, false, pa, ptr(probs16), int64(m*n), va, v16, int64(n*e), oa, int64(m*e), count, 0)
	})
	parents := []*Tensor{q, k, v}
	if mask != nil {
		parents = append(parents, mask)
	}
	r := resultGPU(out, outShape, parents, func(g *cuda.Buffer) {
		gp := ptr(g)
		var g16 *cuda.Buffer
		if options.BF16Autocast {
			g16 = castBF16(gp, g.Size()/4)
			defer g16.Free()
		}
		if dv := v.ensureGradGPU(); dv != nil {
			vp := ptr(dv)
			timedGPU("attn_dv", func() error {
				return options.gemmBatchedPrepared(n, e, m, true, false, pa, ptr(probs16), int64(m*n), gp, ptr(g16), int64(m*e), vp, int64(n*e), count, 1)
			})
		}
		if !q.RequiresGrad && !k.RequiresGrad && (mask == nil || !mask.RequiresGrad) {
			return
		}
		dp, ds := mustAlloc(count*m*n), mustAlloc(count*m*n)
		defer dp.Free()
		defer ds.Free()
		dpp, dsp := ptr(dp), ptr(ds)
		timedGPU("attn_dp", func() error {
			return options.gemmBatchedPrepared(m, n, e, false, true, gp, ptr(g16), int64(m*e), va, v16, int64(n*e), dpp, int64(m*n), count, 0)
		})
		var dmask uintptr
		if mask != nil {
			dmask = ptr(mask.ensureGradGPU())
		}
		launch("attn_softmax_b", int(rows)*32, unsafe.Pointer(&pa), unsafe.Pointer(&dpp), unsafe.Pointer(&dsp), unsafe.Pointer(&dmask), unsafe.Pointer(&rows), unsafe.Pointer(&dim), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
		var ds16 *cuda.Buffer
		if options.BF16Autocast {
			ds16 = castBF16(dsp, count*m*n)
			defer ds16.Free()
		}
		if dq := q.ensureGradGPU(); dq != nil {
			qp := ptr(dq)
			timedGPU("attn_dq", func() error {
				return options.gemmBatchedPrepared(m, d, n, false, false, dsp, ptr(ds16), int64(m*n), ka, k16, int64(n*d), qp, int64(m*d), count, 1)
			})
		}
		if dk := k.ensureGradGPU(); dk != nil {
			kp := ptr(dk)
			timedGPU("attn_dk", func() error {
				return options.gemmBatchedPrepared(n, d, m, true, false, dsp, ptr(ds16), int64(m*n), qa, q16, int64(m*d), kp, int64(n*d), count, 1)
			})
		}
	})
	r.aux = []*cuda.Buffer{probs}
	if probs16 != nil {
		r.aux = append(r.aux, probs16)
	}
	return r
}

func canFuseAttention(q, k, v, mask *Tensor) bool {
	if len(q.Shape) < 2 || len(q.Shape) != len(k.Shape) || len(q.Shape) != len(v.Shape) || len(q.Shape) > 6 {
		return false
	}
	for i := 0; i < len(q.Shape)-2; i++ {
		if q.Shape[i] != k.Shape[i] || q.Shape[i] != v.Shape[i] {
			return false
		}
	}
	if q.Shape[len(q.Shape)-1] != k.Shape[len(k.Shape)-1] || k.Shape[len(k.Shape)-2] != v.Shape[len(v.Shape)-2] {
		return false
	}
	if mask != nil {
		same(q, mask)
		scores := append(append([]int(nil), q.Shape[:len(q.Shape)-2]...), q.Shape[len(q.Shape)-2], k.Shape[len(k.Shape)-2])
		broadcast := bshape(scores, mask.Shape)
		if len(broadcast) != len(scores) {
			return false
		}
		for i := range scores {
			if broadcast[i] != scores[i] {
				return false
			}
		}
	}
	return true
}
