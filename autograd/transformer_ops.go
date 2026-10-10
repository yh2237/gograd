package autograd

import (
	"math"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

func axisParts(shape []int, axis int) (int, int, int) {
	if axis < 0 {
		axis += len(shape)
	}
	if axis < 0 || axis >= len(shape) {
		panic("autograd: invalid axis")
	}
	return numel(shape[:axis]), shape[axis], numel(shape[axis+1:])
}

func Softmax(a *Tensor, axis int) *Tensor {
	dispatchBackend("softmax", a.Device)
	return softmax(a, axis, false)
}
func LogSoftmax(a *Tensor, axis int) *Tensor {
	dispatchBackend("log_softmax", a.Device)
	return softmax(a, axis, true)
}
func softmax(a *Tensor, axis int, logmode bool) *Tensor {
	name := "softmax"
	if logmode {
		name = "log_softmax"
	}
	useCUDA := dispatchBackend(name, a.Device)
	a = a.Contiguous()
	outer, dim, inner := axisParts(a.Shape, axis)
	if dim == 0 {
		panic("autograd: softmax empty axis")
	}
	if useCUDA {
		out := mustAlloc(a.Numel())
		ap, yp := ptr(a.buf), ptr(out)
		rows, d, in, mode := int32(outer*inner), int32(dim), int32(inner), int32(0)
		if logmode {
			mode = 1
		}
		launch("softmax_f", int(rows)*32, unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&rows), unsafe.Pointer(&d), unsafe.Pointer(&in), unsafe.Pointer(&mode))
		return resultGPU(out, a.Shape, []*Tensor{a}, func(g *cuda.Buffer) {
			if dx := a.ensureGradGPU(); dx != nil {
				gp, dp := ptr(g), ptr(dx)
				launch("softmax_b", int(rows)*32, unsafe.Pointer(&yp), unsafe.Pointer(&gp), unsafe.Pointer(&dp), unsafe.Pointer(&rows), unsafe.Pointer(&d), unsafe.Pointer(&in), unsafe.Pointer(&mode))
			}
		})
	}
	v := cpuAlloc(a.Numel())
	parallelFor(outer*inner, func(start, end int) {
		for row := start; row < end; row++ {
			base := (row/inner)*dim*inner + row%inner
			mx := float32(math.Inf(-1))
			for k := 0; k < dim; k++ {
				mx = max(mx, a.Data[base+k*inner])
			}
			var sum float32
			for k := 0; k < dim; k++ {
				sum += float32(math.Exp(float64(a.Data[base+k*inner] - mx)))
			}
			ls := float32(math.Log(float64(sum)))
			for k := 0; k < dim; k++ {
				idx := base + k*inner
				if logmode {
					v[idx] = a.Data[idx] - mx - ls
				} else {
					v[idx] = float32(math.Exp(float64(a.Data[idx]-mx))) / sum
				}
			}
		}
	})
	return result(v, a.Shape, []*Tensor{a}, func(g []float32) {
		dx := cpuAlloc(a.Numel())
		parallelFor(outer*inner, func(start, end int) {
			for row := start; row < end; row++ {
				base := (row/inner)*dim*inner + row%inner
				var dot float32
				for k := 0; k < dim; k++ {
					idx := base + k*inner
					if logmode {
						dot += g[idx]
					} else {
						dot += g[idx] * v[idx]
					}
				}
				for k := 0; k < dim; k++ {
					idx := base + k*inner
					if logmode {
						dx[idx] = g[idx] - float32(math.Exp(float64(v[idx])))*dot
					} else {
						dx[idx] = v[idx] * (g[idx] - dot)
					}
				}
			}
		})
		a.addGrad(dx)
		cpuRelease(dx)
	})
}

// Dropout uses the same index/seed hash on CPU and CUDA. A fixed seed makes a
// forward/backward pair reproducible without keeping a mask tensor.
func Dropout(a *Tensor, p float32, seed uint32, training bool) *Tensor {
	useCUDA := dispatchBackend("dropout", a.Device)
	if !training || p == 0 {
		return a
	}
	if p < 0 || p >= 1 {
		panic("autograd: dropout probability")
	}
	a = a.Contiguous()
	cutoff := uint32(float64(p) * 4294967296.0)
	if useCUDA {
		n := int32(a.Numel())
		out := mustAlloc(int(n))
		ap, yp := ptr(a.buf), ptr(out)
		launch("dropout_f", int(n), unsafe.Pointer(&ap), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&p), unsafe.Pointer(&seed), unsafe.Pointer(&cutoff))
		return resultGPU(out, a.Shape, []*Tensor{a}, func(g *cuda.Buffer) {
			if dx := a.ensureGradGPU(); dx != nil {
				gp, dp := ptr(g), ptr(dx)
				launch("dropout_b", int(n), unsafe.Pointer(&gp), unsafe.Pointer(&dp), unsafe.Pointer(&n), unsafe.Pointer(&p), unsafe.Pointer(&seed), unsafe.Pointer(&cutoff))
			}
		})
	}
	v := cpuAlloc(a.Numel())
	keep := func(i int) bool {
		x := uint32(i) ^ seed
		x ^= x >> 16
		x *= 0x7feb352d
		x ^= x >> 15
		x *= 0x846ca68b
		x ^= x >> 16
		return x >= cutoff
	}
	parallelFor(len(v), func(start, end int) {
		for i := start; i < end; i++ {
			if keep(i) {
				v[i] = a.Data[i] / (1 - p)
			}
		}
	})
	return result(v, a.Shape, []*Tensor{a}, func(g []float32) {
		dx := cpuAlloc(len(g))
		parallelFor(len(g), func(start, end int) {
			for i := start; i < end; i++ {
				if keep(i) {
					dx[i] = g[i] / (1 - p)
				}
			}
		})
		a.addGrad(dx)
		cpuRelease(dx)
	})
}

// ScaledDotProductAttention accepts [..., query, depth] and [..., key, depth].
// The additive mask broadcasts over the attention score shape.
// AttentionAlgorithm is the compatibility policy for unbound tensors:
// "materialized", "flash", or "auto". Explicit contexts use ExecutionOptions.
// Auto selects the tiled path only
// when materialized score scratch would exceed roughly 1 GiB.
var AttentionAlgorithm = "auto"

func ScaledDotProductAttention(q, k, v, mask *Tensor) *Tensor {
	dispatchBackend("attention", q.Device)
	return attentionWithDropout(q, k, v, mask, 0, 0, false)
}
func attentionWithDropout(q, k, v, mask *Tensor, p float32, seed uint32, training bool) *Tensor {
	useCUDA := dispatchBackend("attention", q.Device)
	same(q, k, v, mask)
	options := executionOptions(q, k, v, mask)
	if len(q.Shape) < 2 || len(k.Shape) < 2 || len(v.Shape) < 2 {
		panic("autograd: attention rank")
	}
	depth := q.Shape[len(q.Shape)-1]
	if depth != k.Shape[len(k.Shape)-1] {
		panic("autograd: attention depth")
	}
	if useCUDA && (!training || p == 0) && canFuseAttention(q, k, v, mask) {
		flashSupported := q.Shape[len(q.Shape)-1] <= 128 && v.Shape[len(v.Shape)-1] <= 128
		switch options.AttentionAlgorithm {
		case "flash":
			if !flashSupported {
				panic("autograd: flash attention head dimension exceeds 128")
			}
			return gpuFlashAttention(q, k, v, mask)
		case "auto":
			batchCount := numel(q.Shape[:len(q.Shape)-2])
			if flashSupported && int64(batchCount)*int64(q.Shape[len(q.Shape)-2])*int64(k.Shape[len(k.Shape)-2])*12 >= 1<<30 {
				return gpuFlashAttention(q, k, v, mask)
			}
		case "materialized":
		default:
			panic("autograd: unknown attention algorithm")
		}
		return gpuAttention(q, k, v, mask, options)
	}
	scores := MulScalar(MatMul(q, Transpose(k, len(k.Shape)-2, len(k.Shape)-1)), 1/float32(math.Sqrt(float64(depth))))
	if mask != nil {
		scores = Add(scores, mask)
	}
	weights := Dropout(Softmax(scores, -1), p, seed, training)
	return MatMul(weights, v)
}

// CrossEntropy treats the final logits axis as classes and averages over
// targets other than ignoreIndex. Targets are integer class IDs on the host.
func CrossEntropy(logits *Tensor, targets []int, ignoreIndex int) *Tensor {
	useCUDA := dispatchBackend("cross_entropy", logits.Device)
	logits = logits.Contiguous()
	if len(logits.Shape) < 1 {
		panic("autograd: cross entropy rank")
	}
	classes := logits.Shape[len(logits.Shape)-1]
	if classes < 1 {
		panic("autograd: cross entropy empty class axis")
	}
	rows := logits.Numel() / classes
	if len(targets) != rows {
		panic("autograd: cross entropy target shape")
	}
	for _, id := range targets {
		if id != ignoreIndex && (id < 0 || id >= classes) {
			panic("autograd: cross entropy target index")
		}
	}
	if useCUDA {
		ids := make([]int32, rows)
		for i, id := range targets {
			ids[i] = int32(id)
		}
		idBuf := mustAlloc(rows)
		if err := idBuf.CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&ids[0])), rows*4)); err != nil {
			panic(err)
		}
		acc := mustAlloc(2)
		if err := acc.Memset(0, acc.Size()); err != nil {
			panic(err)
		}
		out := mustAlloc(1)
		xp, ip, ap, yp := ptr(logits.buf), ptr(idBuf), ptr(acc), ptr(out)
		r, c, ig := int32(rows), int32(classes), int32(ignoreIndex)
		launch("ce_f", rows, unsafe.Pointer(&xp), unsafe.Pointer(&ip), unsafe.Pointer(&ap), unsafe.Pointer(&r), unsafe.Pointer(&c), unsafe.Pointer(&ig))
		launch("ce_finish", 1, unsafe.Pointer(&ap), unsafe.Pointer(&yp))
		res := resultGPU(out, []int{}, []*Tensor{logits}, func(g *cuda.Buffer) {
			if dx := logits.ensureGradGPU(); dx != nil {
				gp, dp := ptr(g), ptr(dx)
				launch("ce_b", rows, unsafe.Pointer(&xp), unsafe.Pointer(&ip), unsafe.Pointer(&ap), unsafe.Pointer(&gp), unsafe.Pointer(&dp), unsafe.Pointer(&r), unsafe.Pointer(&c), unsafe.Pointer(&ig))
			}
		})
		res.aux = []*cuda.Buffer{idBuf, acc}
		return res
	}
	count := 0
	var loss float32
	for row, id := range targets {
		if id == ignoreIndex {
			continue
		}
		base := row * classes
		mx := float32(math.Inf(-1))
		for j := 0; j < classes; j++ {
			mx = max(mx, logits.Data[base+j])
		}
		var sum float32
		for j := 0; j < classes; j++ {
			sum += float32(math.Exp(float64(logits.Data[base+j] - mx)))
		}
		loss += mx + float32(math.Log(float64(sum))) - logits.Data[base+id]
		count++
	}
	if count > 0 {
		loss /= float32(count)
	} else {
		loss = float32(math.NaN())
	}
	v := cpuAlloc(1)
	v[0] = loss
	return result(v, []int{}, []*Tensor{logits}, func(g []float32) {
		dx := cpuAlloc(logits.Numel())
		if count > 0 {
			for row, id := range targets {
				if id == ignoreIndex {
					continue
				}
				base := row * classes
				mx := float32(math.Inf(-1))
				for j := 0; j < classes; j++ {
					mx = max(mx, logits.Data[base+j])
				}
				var sum float32
				for j := 0; j < classes; j++ {
					sum += float32(math.Exp(float64(logits.Data[base+j] - mx)))
				}
				for j := 0; j < classes; j++ {
					dx[base+j] = g[0] / float32(count) * float32(math.Exp(float64(logits.Data[base+j]-mx))) / sum
				}
				dx[base+id] -= g[0] / float32(count)
			}
		}
		logits.addGrad(dx)
		cpuRelease(dx)
	})
}
