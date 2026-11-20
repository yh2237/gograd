package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"unsafe"
)

func mustAlloc(n int) *cuda.Buffer {
	b, e := cuda.Alloc(n * 4)
	if e != nil {
		panic(e)
	}
	return b
}
func viewMeta(shape, stride []int) ([6]int32, [6]int32) {
	if len(shape) > 6 {
		panic("autograd: CUDA view rank > 6")
	}
	s := [6]int32{1, 1, 1, 1, 1, 1}
	var st [6]int32
	for i, d := range shape {
		s[6-len(shape)+i] = int32(d)
		st[6-len(shape)+i] = int32(stride[i])
	}
	return s, st
}
func gpuViewGather(src, dst *cuda.Buffer, shape, stride []int, offset int) {
	s, st := viewMeta(shape, stride)
	n, o := int32(numel(shape)), int32(offset)
	a, b := ptr(src), ptr(dst)
	launch("view_gather", int(n), unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&n), unsafe.Pointer(&s), unsafe.Pointer(&st), unsafe.Pointer(&o))
}
func gpuViewGatherBF16(src, dst *cuda.Buffer, shape, stride []int, offset int) {
	s, st := viewMeta(shape, stride)
	n, o := int32(numel(shape)), int32(offset)
	a, b := ptr(src), ptr(dst)
	launch("view_gather_bf16", int(n), unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&n), unsafe.Pointer(&s), unsafe.Pointer(&st), unsafe.Pointer(&o))
}
func gpuViewScatter(src, dst *cuda.Buffer, shape, stride []int, offset int) {
	s, st := viewMeta(shape, stride)
	n, o := int32(numel(shape)), int32(offset)
	a, b := ptr(src), ptr(dst)
	launch("view_scatter", int(n), unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&n), unsafe.Pointer(&s), unsafe.Pointer(&st), unsafe.Pointer(&o))
}
func gpuScalar(a *Tensor, value float32, op int) *Tensor {
	n := int32(a.Numel())
	out := mustAlloc(int(n))
	ap, yp := ptr(a.buf), ptr(out)
	kind := int32(op)
	launch("scalar_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&value), unsafe.Pointer(&kind))
	return resultGPU(out, a.Shape, []*Tensor{a}, func(g *cuda.Buffer) {
		da := ptr(a.ensureGradGPU())
		if da == 0 {
			return
		}
		gp := ptr(g)
		launch("scalar_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&da), unsafe.Pointer(&n), unsafe.Pointer(&value), unsafe.Pointer(&kind))
	})
}
func gpuClamp(a *Tensor, lo, hi float32) *Tensor {
	n := int32(a.Numel())
	out := mustAlloc(int(n))
	ap, yp := ptr(a.buf), ptr(out)
	launch("clamp_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&lo), unsafe.Pointer(&hi))
	return resultGPU(out, a.Shape, []*Tensor{a}, func(g *cuda.Buffer) {
		da := ptr(a.ensureGradGPU())
		if da == 0 {
			return
		}
		gp := ptr(g)
		launch("clamp_b", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&gp), unsafe.Pointer(&da), unsafe.Pointer(&n), unsafe.Pointer(&lo), unsafe.Pointer(&hi))
	})
}
func gpuPowScalar(a *Tensor, p float32) *Tensor {
	n := int32(a.Numel())
	out := mustAlloc(int(n))
	ap, yp := ptr(a.buf), ptr(out)
	launch("pow_scalar_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&p))
	return resultGPU(out, a.Shape, []*Tensor{a}, func(g *cuda.Buffer) {
		da := ptr(a.ensureGradGPU())
		if da == 0 {
			return
		}
		gp := ptr(g)
		launch("pow_scalar_b", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&gp), unsafe.Pointer(&da), unsafe.Pointer(&n), unsafe.Pointer(&p))
	})
}
func aligned3(shape []int) [3]int32 {
	if len(shape) > 3 {
		panic("autograd: CUDA rank > 3")
	}
	out := [3]int32{1, 1, 1}
	for i, v := range shape {
		out[3-len(shape)+i] = int32(v)
	}
	return out
}
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func isLastDimBias(a, b []int) bool {
	if len(a) < 2 || len(b) == 0 || len(b) > len(a) {
		return false
	}
	cols := a[len(a)-1]
	if cols <= 0 || b[len(b)-1] != cols {
		return false
	}
	for i := 0; i < len(b)-1; i++ {
		if b[i] != 1 {
			return false
		}
	}
	return true
}
func gpuBinary(a, b *Tensor, op int) *Tensor {
	options := executionOptions(a, b)
	s := bshape(a.Shape, b.Shape)
	if sameShape(a.Shape, b.Shape) {
		n := numel(s)
		out := mustAlloc(n)
		var shadow *cuda.Buffer
		if options.BF16Autocast {
			shadow = allocBF16(n)
		}
		ap, bp, yp := ptr(a.buf), ptr(b.buf), ptr(out)
		n32, ao := int32(n), int32(op)
		if options.BF16Autocast {
			sh := ptr(shadow)
			launch("binary_same_bf16_f", n, unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n32), unsafe.Pointer(&ao))
		} else {
			launch("binary_same_f", n, unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n32), unsafe.Pointer(&ao))
		}
		res := resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
			if n == 0 {
				a.ensureGradGPU()
				b.ensureGradGPU()
				return
			}
			da, betaA := a.allocGradGPU()
			db, betaB := b.allocGradGPU()
			ag, bg, gp := ptr(da), ptr(db), ptr(g)
			addA, addB := int32(0), int32(0)
			if betaA != 0 {
				addA = 1
			}
			if betaB != 0 {
				addB = 1
			}
			launch("binary_same_b", n, unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&gp), unsafe.Pointer(&ag), unsafe.Pointer(&bg), unsafe.Pointer(&n32), unsafe.Pointer(&ao), unsafe.Pointer(&addA), unsafe.Pointer(&addB))
		})
		res.setBF16(shadow)
		return res
	}
	if op == 0 && isLastDimBias(a.Shape, b.Shape) {
		n := numel(s)
		cols := a.Shape[len(a.Shape)-1]
		rows := n / cols
		out := mustAlloc(n)
		var shadow *cuda.Buffer
		if options.BF16Autocast {
			shadow = allocBF16(n)
		}
		ap, bp, yp := ptr(a.buf), ptr(b.buf), ptr(out)
		n32, c32, r32 := int32(n), int32(cols), int32(rows)
		if options.BF16Autocast {
			sh := ptr(shadow)
			launch("bias_add_bf16_f", n, unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n32), unsafe.Pointer(&c32))
		} else {
			launch("bias_add_f", n, unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n32), unsafe.Pointer(&c32))
		}
		res := resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
			if n == 0 {
				a.ensureGradGPU()
				b.ensureGradGPU()
				return
			}
			if a.RequiresGrad {
				a.copyOrAddGradGPU(g, n)
			}
			if db := b.ensureGradGPU(); db != nil {
				gp, bg := ptr(g), ptr(db)
				gy := min(16, (rows+63)/64)
				if gy < 1 {
					gy = 1
				}
				timedGPU("bias_add_b", func() error {
					return graphKernel("bias_add_b").Launch([3]int{(cols + 31) / 32, gy, 1}, [3]int{32, 8, 1}, 0, nil, []unsafe.Pointer{unsafe.Pointer(&gp), unsafe.Pointer(&bg), unsafe.Pointer(&r32), unsafe.Pointer(&c32)})
				})
			}
		})
		res.setBF16(shadow)
		return res
	}
	if len(s) > 3 {
		return gpuBinaryND(a, b, s, op)
	}
	out := mustAlloc(numel(s))
	var shadow *cuda.Buffer
	if options.BF16Autocast {
		shadow = allocBF16(numel(s))
	}
	ad, bd := aligned3(a.Shape), aligned3(b.Shape)
	od := aligned3(s)
	ap, bp, yp := ptr(a.buf), ptr(b.buf), ptr(out)
	n, d1, d2 := int32(numel(s)), od[1], od[2]
	ao, bo := int32(op), int32(op)
	_ = bo
	if options.BF16Autocast {
		sh := ptr(shadow)
		launch("binary_bf16_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n), unsafe.Pointer(&d1), unsafe.Pointer(&d2), unsafe.Pointer(&ad[0]), unsafe.Pointer(&ad[1]), unsafe.Pointer(&ad[2]), unsafe.Pointer(&bd[0]), unsafe.Pointer(&bd[1]), unsafe.Pointer(&bd[2]), unsafe.Pointer(&ao))
	} else {
		launch("binary_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&d1), unsafe.Pointer(&d2), unsafe.Pointer(&ad[0]), unsafe.Pointer(&ad[1]), unsafe.Pointer(&ad[2]), unsafe.Pointer(&bd[0]), unsafe.Pointer(&bd[1]), unsafe.Pointer(&bd[2]), unsafe.Pointer(&ao))
	}
	res := resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		ag, bg := ptr(a.ensureGradGPU()), ptr(b.ensureGradGPU())
		gp := ptr(g)
		launch("binary_b", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&gp), unsafe.Pointer(&ag), unsafe.Pointer(&bg), unsafe.Pointer(&n), unsafe.Pointer(&d1), unsafe.Pointer(&d2), unsafe.Pointer(&ad[0]), unsafe.Pointer(&ad[1]), unsafe.Pointer(&ad[2]), unsafe.Pointer(&bd[0]), unsafe.Pointer(&bd[1]), unsafe.Pointer(&bd[2]), unsafe.Pointer(&ao))
	})
	res.setBF16(shadow)
	return res
}
func gpuBinaryND(a, b *Tensor, s []int, op int) *Tensor {
	options := executionOptions(a, b)
	broadcastStride := func(t *Tensor) []int {
		st := make([]int, len(s))
		for i := range s {
			j := i - len(s) + len(t.Shape)
			if j >= 0 && t.Shape[j] != 1 {
				st[i] = t.Strides[j]
			}
		}
		return st
	}
	shape, _ := viewMeta(s, strides(s))
	_, as := viewMeta(s, broadcastStride(a))
	_, bs := viewMeta(s, broadcastStride(b))
	n, kind := int32(numel(s)), int32(op)
	out := mustAlloc(int(n))
	var shadow *cuda.Buffer
	if options.BF16Autocast {
		shadow = allocBF16(int(n))
	}
	ap, bp, yp := ptr(a.buf), ptr(b.buf), ptr(out)
	if options.BF16Autocast {
		sh := ptr(shadow)
		launch("binary_nd_bf16_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sh), unsafe.Pointer(&n), unsafe.Pointer(&shape), unsafe.Pointer(&as), unsafe.Pointer(&bs), unsafe.Pointer(&kind))
	} else {
		launch("binary_nd_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&shape), unsafe.Pointer(&as), unsafe.Pointer(&bs), unsafe.Pointer(&kind))
	}
	res := resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		ag, bg := ptr(a.ensureGradGPU()), ptr(b.ensureGradGPU())
		gp := ptr(g)
		launch("binary_nd_b", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&gp), unsafe.Pointer(&ag), unsafe.Pointer(&bg), unsafe.Pointer(&n), unsafe.Pointer(&shape), unsafe.Pointer(&as), unsafe.Pointer(&bs), unsafe.Pointer(&kind))
	})
	res.setBF16(shadow)
	return res
}
func gpuUnary(a *Tensor, op int) *Tensor {
	n := int32(a.Numel())
	out := mustAlloc(int(n))
	ap, yp := ptr(a.buf), ptr(out)
	kind := int32(op)
	launch("unary_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&kind))
	return resultGPU(out, a.Shape, []*Tensor{a}, func(g *cuda.Buffer) {
		da := ptr(a.ensureGradGPU())
		if da == 0 {
			return
		}
		gp := ptr(g)
		launch("unary_b", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&gp), unsafe.Pointer(&da), unsafe.Pointer(&n), unsafe.Pointer(&kind))
	})
}
func gpuConcat(axis int, a, b *Tensor) *Tensor {
	s := append([]int(nil), a.Shape...)
	s[axis] += b.Shape[axis]
	outer, inner := numel(s[:axis]), numel(s[axis+1:])
	n := numel(s)
	out := mustAlloc(n)
	ap, bp, yp := ptr(a.buf), ptr(b.buf), ptr(out)
	o, ac, bc, in := int32(outer), int32(a.Shape[axis]), int32(b.Shape[axis]), int32(inner)
	launch("concat_f", n, unsafe.Pointer(&ap), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&o), unsafe.Pointer(&ac), unsafe.Pointer(&bc), unsafe.Pointer(&in))
	return resultGPU(out, s, []*Tensor{a, b}, func(g *cuda.Buffer) {
		da, db := ptr(a.ensureGradGPU()), ptr(b.ensureGradGPU())
		gp := ptr(g)
		launch("concat_b", n, unsafe.Pointer(&gp), unsafe.Pointer(&da), unsafe.Pointer(&db), unsafe.Pointer(&o), unsafe.Pointer(&ac), unsafe.Pointer(&bc), unsafe.Pointer(&in))
	})
}
func gpuEmbedding(w *Tensor, ids []int, shape []int, padding int) *Tensor {
	id32 := make([]int32, len(ids))
	for i, v := range ids {
		id32[i] = int32(v)
	}
	idBuf := mustAlloc(len(ids))
	// An empty batch has no index to upload.
	if len(ids) > 0 {
		if e := idBuf.CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&id32[0])), len(ids)*4)); e != nil {
			panic(e)
		}
	}
	return gpuEmbeddingBuffer(w, idBuf, shape, padding, true)
}
func gpuEmbeddingBuffer(w *Tensor, idBuf *cuda.Buffer, shape []int, padding int, own bool) *Tensor {
	dim := w.Shape[1]
	s := append(append([]int(nil), shape...), dim)
	n := int32(numel(shape) * dim)
	out := mustAlloc(int(n))
	wp, ip, yp := ptr(w.buf), ptr(idBuf), ptr(out)
	d, pad := int32(dim), int32(padding)
	launch("embedding_f", int(n), unsafe.Pointer(&wp), unsafe.Pointer(&ip), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&d))
	r := resultGPU(out, s, []*Tensor{w}, func(g *cuda.Buffer) {
		dw := ptr(w.ensureGradGPU())
		if dw == 0 {
			return
		}
		gp := ptr(g)
		launch("embedding_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&ip), unsafe.Pointer(&dw), unsafe.Pointer(&n), unsafe.Pointer(&d), unsafe.Pointer(&pad))
	})
	if own {
		if r.RequiresGrad {
			r.aux = []*cuda.Buffer{idBuf}
		} else {
			idBuf.Free()
		}
	}
	return r
}
func isLastAxisOnly(selected []bool) bool {
	if len(selected) < 2 || !selected[len(selected)-1] {
		return false
	}
	for i := 0; i < len(selected)-1; i++ {
		if selected[i] {
			return false
		}
	}
	return true
}
func gpuReduce(a *Tensor, s []int, selected []bool, mean bool) *Tensor {
	if len(a.Shape) > 6 {
		panic("autograd: CUDA reduction rank > 6")
	}
	outCount := numel(s)
	if a.Numel() == 0 {
		out := mustAlloc(outCount)
		var fill byte
		if mean {
			fill = 0xff // 0xffffffff represents a float32 NaN.
		}
		if err := out.Memset(fill, out.Size()); err != nil {
			out.Free()
			panic(err)
		}
		return resultGPU(out, s, []*Tensor{a}, func(*cuda.Buffer) {
			a.ensureGradGPU()
		})
	}
	if len(a.Shape) > 3 {
		return gpuReduceND(a, s, selected, mean)
	}
	out := mustAlloc(outCount)
	lastOnly := isLastAxisOnly(selected) && outCount > 1
	if !lastOnly {
		if e := out.Memset(0, out.Size()); e != nil {
			panic(e)
		}
	}
	inDims := aligned3(a.Shape)
	outStrides := strides(s)
	var rs [3]int32
	next := 0
	for i := range a.Shape {
		if !selected[i] {
			rs[3-len(a.Shape)+i] = int32(outStrides[next])
			next++
		}
	}
	n, d1, d2 := int32(a.Numel()), inDims[1], inDims[2]
	scale := float32(1)
	if mean {
		scale = 1 / float32(a.Numel()/outCount)
	}
	ap, yp := ptr(a.buf), ptr(out)
	if outCount == 1 {
		launchScalarReduce(ap, yp, n, scale)
	} else if lastOnly {
		rows, dim := int32(outCount), int32(a.Shape[len(a.Shape)-1])
		launch("reduce_last_f", outCount*32, unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&rows), unsafe.Pointer(&dim), unsafe.Pointer(&scale))
	} else {
		launch("reduce_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&d1), unsafe.Pointer(&d2), unsafe.Pointer(&rs[0]), unsafe.Pointer(&rs[1]), unsafe.Pointer(&rs[2]), unsafe.Pointer(&scale))
	}
	return resultGPU(out, s, []*Tensor{a}, func(g *cuda.Buffer) {
		dx := ptr(a.ensureGradGPU())
		if dx == 0 {
			return
		}
		gp := ptr(g)
		launch("reduce_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&n), unsafe.Pointer(&d1), unsafe.Pointer(&d2), unsafe.Pointer(&rs[0]), unsafe.Pointer(&rs[1]), unsafe.Pointer(&rs[2]), unsafe.Pointer(&scale))
	})
}
func gpuReduceND(a *Tensor, s []int, selected []bool, mean bool) *Tensor {
	mapStride := make([]int, len(a.Shape))
	outStride := strides(s)
	next := 0
	for i := range a.Shape {
		if !selected[i] {
			mapStride[i] = outStride[next]
			next++
		}
	}
	shape, mapped := viewMeta(a.Shape, mapStride)
	n := int32(a.Numel())
	outCount := numel(s)
	scale := float32(1)
	if mean {
		scale = 1 / float32(a.Numel()/outCount)
	}
	out := mustAlloc(outCount)
	lastOnly := isLastAxisOnly(selected) && outCount > 1
	if !lastOnly {
		if err := out.Memset(0, out.Size()); err != nil {
			panic(err)
		}
	}
	ap, yp := ptr(a.buf), ptr(out)
	if outCount == 1 {
		launchScalarReduce(ap, yp, n, scale)
	} else if lastOnly {
		rows, dim := int32(outCount), int32(a.Shape[len(a.Shape)-1])
		launch("reduce_last_f", outCount*32, unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&rows), unsafe.Pointer(&dim), unsafe.Pointer(&scale))
	} else {
		launch("reduce_nd_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&shape), unsafe.Pointer(&mapped), unsafe.Pointer(&scale))
	}
	return resultGPU(out, s, []*Tensor{a}, func(g *cuda.Buffer) {
		dx := ptr(a.ensureGradGPU())
		if dx == 0 {
			return
		}
		gp := ptr(g)
		launch("reduce_nd_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&n), unsafe.Pointer(&shape), unsafe.Pointer(&mapped), unsafe.Pointer(&scale))
	})
}
func launchScalarReduce(x, y uintptr, n int32, scale float32) {
	launch("reduce_scalar_f", min(int(n), 256*256), unsafe.Pointer(&x), unsafe.Pointer(&y), unsafe.Pointer(&n), unsafe.Pointer(&scale))
}

func gpuGroupNorm(x, w, b *Tensor, groups int, eps float32) *Tensor {
	s := x.Shape
	bs, t, c := s[0], s[1], s[2]
	stats := mustAlloc(bs * groups * 4)
	out := mustAlloc(x.Numel())
	xp, wp, bp, sp, yp := ptr(x.buf), ptr(w.buf), ptr(b.buf), ptr(stats), ptr(out)
	batch, time, channels, ng, n := int32(bs), int32(t), int32(c), int32(groups), int32(x.Numel())
	launch("group_stats", bs*groups*256, unsafe.Pointer(&xp), unsafe.Pointer(&sp), unsafe.Pointer(&batch), unsafe.Pointer(&time), unsafe.Pointer(&channels), unsafe.Pointer(&ng), unsafe.Pointer(&eps))
	launch("group_f", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&bp), unsafe.Pointer(&sp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&time), unsafe.Pointer(&channels), unsafe.Pointer(&ng))
	r := resultGPU(out, s, []*Tensor{x, w, b}, func(g *cuda.Buffer) {
		gp := ptr(g)
		launch("group_backstats", bs*groups*256, unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&gp), unsafe.Pointer(&sp), unsafe.Pointer(&time), unsafe.Pointer(&channels), unsafe.Pointer(&ng))
		dx, dw, db := ptr(x.ensureGradGPU()), ptr(w.ensureGradGPU()), ptr(b.ensureGradGPU())
		launch("group_b", int(n), unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&gp), unsafe.Pointer(&sp), unsafe.Pointer(&dx), unsafe.Pointer(&dw), unsafe.Pointer(&db), unsafe.Pointer(&n), unsafe.Pointer(&time), unsafe.Pointer(&channels), unsafe.Pointer(&ng))
	})
	if r.RequiresGrad {
		r.aux = []*cuda.Buffer{stats}
	} else {
		stats.Free()
	}
	return r
}
func gpuMaskedLoss(pred, target *Tensor, mse bool) *Tensor {
	n := int32(pred.Numel())
	acc := mustAlloc(2)
	if e := acc.Memset(0, 8); e != nil {
		panic(e)
	}
	out := mustAlloc(1)
	pp, tp, ap, yp := ptr(pred.buf), ptr(target.buf), ptr(acc), ptr(out)
	c := int32(pred.Shape[2])
	kind := int32(0)
	if mse {
		kind = 1
	}
	launch("masked_f", int(n), unsafe.Pointer(&pp), unsafe.Pointer(&tp), unsafe.Pointer(&ap), unsafe.Pointer(&n), unsafe.Pointer(&c), unsafe.Pointer(&kind))
	launch("masked_finish", 1, unsafe.Pointer(&ap), unsafe.Pointer(&yp))
	r := resultGPUWithSaved(out, []int{}, []*Tensor{pred}, []*Tensor{target}, func(g *cuda.Buffer) {
		dx := ptr(pred.ensureGradGPU())
		gp := ptr(g)
		launch("masked_b", int(n), unsafe.Pointer(&pp), unsafe.Pointer(&tp), unsafe.Pointer(&ap), unsafe.Pointer(&gp), unsafe.Pointer(&dx), unsafe.Pointer(&n), unsafe.Pointer(&c), unsafe.Pointer(&kind))
	})
	if r.RequiresGrad {
		r.aux = []*cuda.Buffer{acc}
	} else {
		acc.Free()
	}
	return r
}
