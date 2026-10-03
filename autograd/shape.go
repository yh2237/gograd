package autograd

import (
	"github.com/yh2237/gograd/tensor"
	"math"
	"runtime"
	"sync"
)

func Reshape(a *Tensor, s ...int) *Tensor {
	if numel(s) != a.Numel() {
		panic("autograd: reshape size")
	}
	a = a.Contiguous()
	return makeView(a, s, strides(s), 0)
}
func Permute(a *Tensor, axes ...int) *Tensor {
	n := len(a.Shape)
	if len(axes) != n {
		panic("autograd: permutation rank")
	}
	seen := make([]bool, n)
	s := make([]int, n)
	for i, x := range axes {
		if x < 0 || x >= n || seen[x] {
			panic("autograd: invalid permutation")
		}
		seen[x] = true
		s[i] = a.Shape[x]
	}
	a = a.Contiguous()
	st := make([]int, len(axes))
	for i, axis := range axes {
		st[i] = a.Strides[axis]
	}
	return makeView(a, s, st, 0)
}
func Transpose(a *Tensor, i, j int) *Tensor {
	axes := make([]int, len(a.Shape))
	for k := range axes {
		axes[k] = k
	}
	axes[i], axes[j] = axes[j], axes[i]
	return Permute(a, axes...)
}
func Sum(a *Tensor, axes ...int) *Tensor  { return reduce(a, false, axes...) }
func Mean(a *Tensor, axes ...int) *Tensor { return reduce(a, true, axes...) }
func reduce(a *Tensor, mean bool, axes ...int) *Tensor {
	a = a.Contiguous()
	selected := make([]bool, len(a.Shape))
	for _, ax := range axes {
		if ax < 0 {
			ax += len(selected)
		}
		if ax < 0 || ax >= len(selected) || selected[ax] {
			panic("autograd: invalid axis")
		}
		selected[ax] = true
	}
	s := []int{}
	for i, d := range a.Shape {
		if !selected[i] {
			s = append(s, d)
		}
	}
	if a.Device == tensor.CUDA {
		return gpuReduce(a, s, selected, mean)
	}
	v := cpuAlloc(numel(s))
	mapping := make([]int, len(a.Data))
	count := float32(len(a.Data) / len(v))
	outStrides := strides(s)
	mapAt := func(i int) int {
		j := 0
		shift := len(s) - 1
		for k := len(a.Shape) - 1; k >= 0; k-- {
			if !selected[k] {
				j += (i / a.Strides[k] % a.Shape[k]) * outStrides[shift]
				shift--
			}
		}
		return j
	}
	workers := min(runtime.GOMAXPROCS(0), len(a.Data)/65536)
	if workers >= 2 && len(v) <= 65536 {
		partials := make([][]float32, workers)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			start, end := len(a.Data)*w/workers, len(a.Data)*(w+1)/workers
			partials[w] = make([]float32, len(v))
			wg.Add(1)
			go func(w, start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					j := mapAt(i)
					mapping[i] = j
					partials[w][j] += a.Data[i]
				}
			}(w, start, end)
		}
		wg.Wait()
		for _, p := range partials {
			for i, x := range p {
				v[i] += x
			}
		}
	} else {
		for i, x := range a.Data {
			j := mapAt(i)
			mapping[i] = j
			v[j] += x
		}
	}
	if mean {
		for i := range v {
			v[i] /= count
		}
	}
	return result(v, s, []*Tensor{a}, func(g []float32) {
		dx := make([]float32, len(a.Data))
		parallelFor(len(mapping), func(start, end int) {
			for i := start; i < end; i++ {
				dx[i] = g[mapping[i]]
				if mean {
					dx[i] /= count
				}
			}
		})
		a.addGrad(dx)
	})
}
func Concat(axis int, inputs ...*Tensor) *Tensor {
	if len(inputs) == 0 {
		panic("autograd: empty concat")
	}
	s := append([]int(nil), inputs[0].Shape...)
	if axis < 0 {
		axis += len(s)
	}
	s[axis] = 0
	for _, x := range inputs {
		same(inputs[0], x)
		for i, d := range x.Shape {
			if i != axis && d != s[i] {
				panic("autograd: concat shape")
			}
		}
		s[axis] += x.Shape[axis]
	}
	for i, x := range inputs {
		inputs[i] = x.Contiguous()
	}
	if inputs[0].Device == tensor.CUDA {
		out := inputs[0]
		for _, p := range inputs[1:] {
			out = gpuConcat(axis, out, p)
		}
		return out
	}
	v := cpuAlloc(numel(s))
	outer := numel(s[:axis])
	inner := numel(s[axis+1:])
	for o := 0; o < outer; o++ {
		off := 0
		for _, x := range inputs {
			size := x.Shape[axis] * inner
			copy(v[o*s[axis]*inner+off:], x.Data[o*size:(o+1)*size])
			off += size
		}
	}
	return result(v, s, inputs, func(g []float32) {
		for _, x := range inputs {
			dx := make([]float32, len(x.Data))
			for o := 0; o < outer; o++ {
				off := 0
				for _, p := range inputs {
					if p == x {
						size := x.Shape[axis] * inner
						copy(dx[o*size:(o+1)*size], g[o*s[axis]*inner+off:])
						break
					}
					off += p.Shape[axis] * inner
				}
			}
			x.addGrad(dx)
		}
	})
}
func Slice(a *Tensor, axis, start, end int) *Tensor {
	if axis < 0 {
		axis += len(a.Shape)
	}
	if axis < 0 || axis >= len(a.Shape) || start < 0 || end < start || end > a.Shape[axis] {
		panic("autograd: slice bounds")
	}
	s := append([]int(nil), a.Shape...)
	s[axis] = end - start
	a = a.Contiguous()
	return makeView(a, s, a.Strides, start*a.Strides[axis])
}
func parallelBatches(count int, fn func(int)) {
	workers := min(runtime.GOMAXPROCS(0), count)
	if workers < 2 {
		for z := 0; z < count; z++ {
			fn(z)
		}
		return
	}
	jobs := make(chan int, count)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for z := range jobs {
				fn(z)
			}
		}()
	}
	for z := 0; z < count; z++ {
		jobs <- z
	}
	close(jobs)
	wg.Wait()
}
func MatMul(a, b *Tensor) *Tensor {
	same(a, b)
	a, b = a.Contiguous(), b.Contiguous()
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		panic("autograd: matmul rank")
	}
	m, k := a.Shape[len(a.Shape)-2], a.Shape[len(a.Shape)-1]
	if b.Shape[len(b.Shape)-2] != k {
		panic("autograd: matmul inner dimension")
	}
	n := b.Shape[len(b.Shape)-1]
	batch := bshape(a.Shape[:len(a.Shape)-2], b.Shape[:len(b.Shape)-2])
	s := append(append([]int(nil), batch...), m, n)
	if a.Device == tensor.CUDA {
		return gpuMatMul(a, b, s, m, n, k, batch)
	}
	count := numel(batch)
	aBatch, bBatch := a.Shape[:len(a.Shape)-2], b.Shape[:len(b.Shape)-2]
	flatLinear := count > 1 && numel(aBatch) == count && numel(bBatch) == 1
	v := cpuAlloc(numel(s))
	if flatLinear {
		tensor.SGEMM(v, a.Data, b.Data, count*m, n, k)
	} else {
		parallelBatches(count, func(z int) {
			ai := bindex(z, batch, aBatch) * m * k
			bi := bindex(z, batch, bBatch) * k * n
			tensor.SGEMMOpWorkers(v[z*m*n:(z+1)*m*n], a.Data[ai:ai+m*k], b.Data[bi:bi+k*n], m, n, k, false, false, 1)
		})
	}
	return result(v, s, []*Tensor{a, b}, func(g []float32) {
		da := cpuAlloc(len(a.Data))
		db := cpuAlloc(len(b.Data))
		if flatLinear {
			tensor.SGEMMOp(da, g, b.Data, count*m, k, n, false, true)
			tensor.SGEMMOp(db, a.Data, g, k, n, count*m, true, false)
			a.addGrad(da)
			b.addGrad(db)
			cpuRelease(da)
			cpuRelease(db)
			return
		}
		if count > 1 && numel(aBatch) == count && numel(bBatch) == count {
			parallelBatches(count, func(z int) {
				ai := bindex(z, batch, aBatch) * m * k
				bi := bindex(z, batch, bBatch) * k * n
				gg := g[z*m*n : (z+1)*m*n]
				tensor.SGEMMOpWorkers(da[ai:ai+m*k], gg, b.Data[bi:bi+k*n], m, k, n, false, true, 1)
				tensor.SGEMMOpWorkers(db[bi:bi+k*n], a.Data[ai:ai+m*k], gg, k, n, m, true, false, 1)
			})
			a.addGrad(da)
			b.addGrad(db)
			cpuRelease(da)
			cpuRelease(db)
			return
		}
		tmpA, tmpB := cpuAlloc(m*k), cpuAlloc(k*n)
		for z := 0; z < count; z++ {
			ai := bindex(z, batch, a.Shape[:len(a.Shape)-2]) * m * k
			bi := bindex(z, batch, b.Shape[:len(b.Shape)-2]) * k * n
			gg := g[z*m*n : (z+1)*m*n]
			tensor.SGEMMOp(tmpA, gg, b.Data[bi:bi+k*n], m, k, n, false, true)
			tensor.SGEMMOp(tmpB, a.Data[ai:ai+m*k], gg, k, n, m, true, false)
			for i, v := range tmpA {
				da[ai+i] += v
			}
			for i, v := range tmpB {
				db[bi+i] += v
			}
		}
		a.addGrad(da)
		b.addGrad(db)
		cpuRelease(tmpA)
		cpuRelease(tmpB)
		cpuRelease(da)
		cpuRelease(db)
	})
}
func Embedding(weight *Tensor, ids []int, shape []int) *Tensor {
	return EmbeddingWithPadding(weight, ids, shape, -1)
}

// EmbeddingWithPadding gathers rows and leaves the padding row's gradient zero.
func EmbeddingWithPadding(weight *Tensor, ids []int, shape []int, paddingIdx int) *Tensor {
	weight = weight.Contiguous()
	if len(weight.Shape) != 2 || numel(shape) != len(ids) {
		panic("autograd: embedding shape")
	}
	if paddingIdx < -1 || paddingIdx >= weight.Shape[0] {
		panic("autograd: padding index")
	}
	dim := weight.Shape[1]
	s := append(append([]int(nil), shape...), dim)
	if weight.Device == tensor.CUDA {
		for _, id := range ids {
			if id < 0 || id >= weight.Shape[0] {
				panic("autograd: embedding index")
			}
		}
		return gpuEmbedding(weight, ids, shape, paddingIdx)
	}
	v := cpuAlloc(len(ids) * dim)
	for i, id := range ids {
		if id < 0 || id >= weight.Shape[0] {
			panic("autograd: embedding index")
		}
		copy(v[i*dim:], weight.Data[id*dim:(id+1)*dim])
	}
	return result(v, s, []*Tensor{weight}, func(g []float32) {
		dw := make([]float32, len(weight.Data))
		for i, id := range ids {
			if id == paddingIdx {
				continue
			}
			for j := 0; j < dim; j++ {
				dw[id*dim+j] += g[i*dim+j]
			}
		}
		weight.addGrad(dw)
	})
}
func MaskedLoss(pred, target *Tensor, mse bool) *Tensor {
	same(pred, target)
	pred, target = pred.Contiguous(), target.Contiguous()
	if len(pred.Shape) != 3 || numel(pred.Shape) != target.Numel() {
		panic("autograd: loss shape")
	}
	if pred.Device == tensor.CUDA {
		return gpuMaskedLoss(pred, target, mse)
	}
	c := pred.Shape[2]
	g := make([]float32, len(pred.Data))
	var total float32
	count := 0
	for row := 0; row < len(g)/c; row++ {
		if math.IsNaN(float64(target.Data[row*c])) {
			continue
		}
		for j := 0; j < c; j++ {
			targetValue := target.Data[row*c+j]
			if math.IsNaN(float64(targetValue)) {
				targetValue = 0
			}
			d := pred.Data[row*c+j] - targetValue
			if mse {
				total += d * d
				g[row*c+j] = 2 * d
			} else {
				total += float32(math.Abs(float64(d)))
				if d > 0 {
					g[row*c+j] = 1
				} else if d < 0 {
					g[row*c+j] = -1
				}
			}
			count++
		}
	}
	if count > 0 {
		total /= float32(count)
		for i := range g {
			g[i] /= float32(count)
		}
	}
	value := cpuAlloc(1)
	value[0] = total
	return result(value, []int{}, []*Tensor{pred}, func(up []float32) {
		dx := make([]float32, len(g))
		for i := range g {
			dx[i] = g[i] * up[0]
		}
		pred.addGrad(dx)
	})
}
