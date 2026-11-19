package autograd

import (
	"math"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

func launchFlash(profileName, kernelName string, blocks, sharedBytes int, args ...unsafe.Pointer) {
	if blocks <= 0 {
		return
	}
	timedGPU(profileName, func() error {
		return graphKernel(kernelName).Launch([3]int{blocks, 1, 1}, [3]int{128, 1, 1}, sharedBytes, nil, args)
	})
}

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
	scale := float32(1 / math.Sqrt(float64(d)))
	dp, ep, kp := d|1, e|1, max(d, e)|1
	small := d <= 64 && e <= 64
	fwdKernel, bwdQKernel, bwdKVKernel := "flash_attention_tiled_f128", "flash_attention_tiled_b128", "flash_attention_tiled_b_kv128"
	fwdBR, bwdQBR, bwdKVBK := 32, 16, 16
	fwdShared := (32*dp + 32*kp + 32*33 + 96) * 4
	bwdQShared := (16*dp + 16*ep + 32*kp + 16*33 + 48) * 4
	bwdKVShared := (32*dp + 32*ep + 32*17 + 48) * 4
	if small {
		fwdKernel, bwdQKernel, bwdKVKernel = "flash_attention_tiled_f", "flash_attention_tiled_b", "flash_attention_tiled_b_kv"
		fwdBR, bwdQBR, bwdKVBK = 32, 32, 32
		fwdShared = (32*dp + 64*kp + 32*65 + 96) * 4
		bwdQShared = (32*dp + 32*ep + 64*kp + 32*65 + 96) * 4
		bwdKVShared = (64*dp + 64*ep + 64*33 + 96) * 4
	}
	fwdQBlocks := int32((nq + fwdBR - 1) / fwdBR)
	bwdQBlocks := int32((nq + bwdQBR - 1) / bwdQBR)
	kvBlocks := int32((nk + bwdKVBK - 1) / bwdKVBK)
	if rows > 0 && nk > 0 && e > 0 {
		launchFlash("flash_attention_tiled_f", fwdKernel, numel(batch)*int(fwdQBlocks), fwdShared, unsafe.Pointer(&qa), unsafe.Pointer(&ka), unsafe.Pointer(&va), unsafe.Pointer(&maskPtr), unsafe.Pointer(&oa), unsafe.Pointer(&sa), unsafe.Pointer(&n), unsafe.Pointer(&m), unsafe.Pointer(&depth), unsafe.Pointer(&width), unsafe.Pointer(&fwdQBlocks), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
	} else if rows > 0 && e > 0 {
		if err := out.Memset(0, out.Size()); err != nil {
			panic(err)
		}
	}
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
		if rows == 0 || nk == 0 || e == 0 {
			return
		}
		deltaBuf := mustAlloc(rows)
		defer deltaBuf.Free()
		dpPtr := ptr(deltaBuf)
		r32 := int32(rows)
		launch("flash_attention_delta", rows*32, unsafe.Pointer(&oa), unsafe.Pointer(&gp), unsafe.Pointer(&dpPtr), unsafe.Pointer(&r32), unsafe.Pointer(&width))
		if dq != 0 || dm != 0 {
			launchFlash("flash_attention_tiled_b", bwdQKernel, numel(batch)*int(bwdQBlocks), bwdQShared, unsafe.Pointer(&qa), unsafe.Pointer(&ka), unsafe.Pointer(&va), unsafe.Pointer(&maskPtr), unsafe.Pointer(&sa), unsafe.Pointer(&dpPtr), unsafe.Pointer(&gp), unsafe.Pointer(&dq), unsafe.Pointer(&dm), unsafe.Pointer(&n), unsafe.Pointer(&m), unsafe.Pointer(&depth), unsafe.Pointer(&width), unsafe.Pointer(&bwdQBlocks), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
		}
		if dk != 0 || dv != 0 {
			launchFlash("flash_attention_tiled_b_kv", bwdKVKernel, numel(batch)*int(kvBlocks), bwdKVShared, unsafe.Pointer(&qa), unsafe.Pointer(&ka), unsafe.Pointer(&va), unsafe.Pointer(&maskPtr), unsafe.Pointer(&sa), unsafe.Pointer(&dpPtr), unsafe.Pointer(&gp), unsafe.Pointer(&dk), unsafe.Pointer(&dv), unsafe.Pointer(&n), unsafe.Pointer(&m), unsafe.Pointer(&depth), unsafe.Pointer(&width), unsafe.Pointer(&kvBlocks), unsafe.Pointer(&scale), unsafe.Pointer(&shape), unsafe.Pointer(&ms))
		}
	})
	if result.RequiresGrad {
		result.aux = []*cuda.Buffer{stats}
	} else {
		stats.Free()
	}
	return result
}
