package autograd

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

// Unsqueeze inserts a size-1 axis at axis, which may equal the rank.
func Unsqueeze(a *Tensor, axis int) *Tensor {
	dispatchBackend("reshape", a.Device)
	if axis < 0 || axis > len(a.Shape) {
		panic(fmt.Errorf("autograd: unsqueeze axis %d out of range for rank %d", axis, len(a.Shape)))
	}
	shape := make([]int, 0, len(a.Shape)+1)
	shape = append(shape, a.Shape[:axis]...)
	shape = append(shape, 1)
	shape = append(shape, a.Shape[axis:]...)
	return Reshape(a, shape...)
}

// Flatten collapses the axes from start through end, both inclusive. Negative
// indices count from the end, and start == end leaves the shape unchanged. A
// scalar becomes shape [1], matching PyTorch.
func Flatten(a *Tensor, start, end int) *Tensor {
	dispatchBackend("reshape", a.Device)
	rank := len(a.Shape)
	if rank == 0 {
		if start != 0 || (end != 0 && end != -1) {
			panic(fmt.Errorf("autograd: flatten range %d..%d out of range for rank 0", start, end))
		}
		return Reshape(a, 1)
	}
	if start < 0 {
		start += rank
	}
	if end < 0 {
		end += rank
	}
	if start < 0 || start >= rank || end < start || end >= rank {
		panic(fmt.Errorf("autograd: flatten range %d..%d out of range for rank %d", start, end, rank))
	}
	if start == end {
		return Reshape(a, a.Shape...)
	}
	size := 1
	for _, d := range a.Shape[start : end+1] {
		size *= d
	}
	shape := make([]int, 0, rank-(end-start))
	shape = append(shape, a.Shape[:start]...)
	shape = append(shape, size)
	shape = append(shape, a.Shape[end+1:]...)
	return Reshape(a, shape...)
}

// Squeeze removes every size-1 axis. A tensor whose axes are all 1 becomes a
// scalar.
func Squeeze(a *Tensor) *Tensor {
	dispatchBackend("reshape", a.Device)
	shape := make([]int, 0, len(a.Shape))
	for _, d := range a.Shape {
		if d != 1 {
			shape = append(shape, d)
		}
	}
	return Reshape(a, shape...)
}

func Reshape(a *Tensor, s ...int) *Tensor {
	dispatchBackend("reshape", a.Device)
	if err := validateShape(s); err != nil {
		panic(err)
	}
	if inferenceDevice(a.Device) {
		inferenceOperands(a)
		if len(s) > 6 {
			panic(gpu.ErrInvalidArgument)
		}
		n, err := gpu.Elements(s...)
		if err != nil {
			panic(err)
		}
		if n != a.Numel() {
			panic("autograd: reshape size")
		}
		st := strides(s)
		for _, stride := range st {
			if _, err := gpu.Elements(stride); err != nil {
				panic(err)
			}
		}
		if !inferenceView(a).Compact() {
			a = a.Contiguous()
		}
		return makeView(a, s, st, a.Offset)
	}
	if numel(s) != a.Numel() {
		panic("autograd: reshape size")
	}
	a = a.Contiguous()
	return makeView(a, s, strides(s), 0)
}
func Permute(a *Tensor, axes ...int) *Tensor {
	dispatchBackend("permute", a.Device)
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
	if !inferenceDevice(a.Device) {
		a = a.Contiguous()
	}
	st := make([]int, len(axes))
	for i, axis := range axes {
		st[i] = a.Strides[axis]
	}
	offset := 0
	if inferenceDevice(a.Device) {
		offset = a.Offset
	}
	return makeView(a, s, st, offset)
}
func Transpose(a *Tensor, i, j int) *Tensor {
	dispatchBackend("transpose", a.Device)
	// Negative axes count from the end, like the other shape operations.
	if i < 0 {
		i += len(a.Shape)
	}
	if j < 0 {
		j += len(a.Shape)
	}
	if i < 0 || i >= len(a.Shape) || j < 0 || j >= len(a.Shape) {
		panic("autograd: invalid permutation")
	}
	axes := make([]int, len(a.Shape))
	for k := range axes {
		axes[k] = k
	}
	axes[i], axes[j] = axes[j], axes[i]
	return Permute(a, axes...)
}
func Sum(a *Tensor, axes ...int) *Tensor {
	if inferenceDevice(a.Device) {
		return inferenceReduce(gpu.ReduceSum, "sum", a, axes...)
	}
	dispatchBackend("sum", a.Device)
	return reduce(a, false, axes...)
}
func Mean(a *Tensor, axes ...int) *Tensor {
	if inferenceDevice(a.Device) {
		return inferenceReduce(gpu.ReduceMean, "mean", a, axes...)
	}
	dispatchBackend("mean", a.Device)
	return reduce(a, true, axes...)
}
func reduce(a *Tensor, mean bool, axes ...int) *Tensor {
	useCUDA := dispatchBackend(map[bool]string{true: "mean", false: "sum"}[mean], a.Device)
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
	if useCUDA {
		return gpuReduce(a, s, selected, mean)
	}
	v := cpuAlloc(numel(s))
	// The divisor is the number of reduced elements per output element. An
	// empty reduction has no output element, so the divisor stays zero instead
	// of dividing by zero.
	divisor := 0
	if len(v) > 0 {
		divisor = len(a.Data) / len(v)
	}
	count := float32(divisor)
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
	// The innermost input dim either survives or is reduced, so one output
	// index covers a whole run of input elements in both cases. Mapping per run
	// keeps the division out of the element loop.
	run := 1
	if len(a.Shape) > 0 {
		run = a.Shape[len(a.Shape)-1]
	}
	innerKept := len(selected) > 0 && !selected[len(selected)-1]
	blocks := len(a.Data) / max(run, 1)
	workers := min(runtime.GOMAXPROCS(0), len(a.Data)/65536)
	if workers >= 2 && len(v) <= 65536 {
		partials := make([][]float32, workers)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			partials[w] = make([]float32, len(v))
			wg.Add(1)
			go func(w, start, end int) {
				defer wg.Done()
				p := partials[w]
				for b := start; b < end; b++ {
					base, j := b*run, mapAt(b*run)
					if innerKept {
						for k := 0; k < run; k++ {
							p[j+k] += a.Data[base+k]
						}
						continue
					}
					var sum float32
					for k := 0; k < run; k++ {
						sum += a.Data[base+k]
					}
					p[j] += sum
				}
			}(w, blocks*w/workers, blocks*(w+1)/workers)
		}
		wg.Wait()
		for _, p := range partials {
			for i, x := range p {
				v[i] += x
			}
		}
	} else {
		for b := 0; b < blocks; b++ {
			base, j := b*run, mapAt(b*run)
			if innerKept {
				for k := 0; k < run; k++ {
					v[j+k] += a.Data[base+k]
				}
				continue
			}
			var sum float32
			for k := 0; k < run; k++ {
				sum += a.Data[base+k]
			}
			v[j] += sum
		}
	}
	if mean {
		for i := range v {
			v[i] /= count
		}
	}
	return result(v, s, []*Tensor{a}, func(g []float32) {
		dx := make([]float32, len(a.Data))
		parallelForWork(blocks, run, func(start, end int) {
			for b := start; b < end; b++ {
				base, j := b*run, mapAt(b*run)
				if innerKept {
					for k := 0; k < run; k++ {
						dx[base+k] = g[j+k]
						if mean {
							dx[base+k] /= count
						}
					}
					continue
				}
				fill := g[j]
				if mean {
					fill /= count
				}
				for k := 0; k < run; k++ {
					dx[base+k] = fill
				}
			}
		})
		a.addGrad(dx)
	})
}

// Stack joins the inputs along a new axis inserted at axis. Every input must
// have the same shape. It composes Unsqueeze and Concat, which own the device
// dispatch.
func Stack(axis int, inputs ...*Tensor) *Tensor {
	if len(inputs) == 0 {
		panic("autograd: empty stack")
	}
	rank := len(inputs[0].Shape)
	if axis < 0 {
		axis += rank + 1
	}
	if axis < 0 || axis > rank {
		panic("autograd: invalid stack axis")
	}
	expanded := make([]*Tensor, len(inputs))
	for i, t := range inputs {
		expanded[i] = Unsqueeze(t, axis)
	}
	return Concat(axis, expanded...)
}

func Concat(axis int, inputs ...*Tensor) *Tensor {
	if len(inputs) == 0 {
		panic("autograd: empty concat")
	}
	if inferenceDevice(inputs[0].Device) {
		return inferenceConcat(axis, inputs...)
	}
	if axis < 0 {
		axis += len(inputs[0].Shape)
	}
	if axis < 0 || axis >= len(inputs[0].Shape) {
		panic("autograd: invalid concat axis")
	}
	// Variadic arguments may alias a caller-owned slice. Keep our compact
	// handles and backward dependencies independent of later caller edits.
	inputs = append([]*Tensor(nil), inputs...)
	same(inputs[0], inputs[1:]...)
	s := append([]int(nil), inputs[0].Shape...)
	s[axis] = 0
	for _, x := range inputs {
		same(inputs[0], x)
		if len(x.Shape) != len(s) {
			panic("autograd: concat shape")
		}
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
	if dispatchBackend("concat", inputs[0].Device) {
		out := inputs[0]
		for _, p := range inputs[1:] {
			if out == p {
				// concat_b writes each occurrence from different threads. A
				// zero-copy identity gives duplicates distinct adjoint buffers,
				// then accumulates them into the shared parent in graph order.
				if _, recording := graphRecording([]*Tensor{p}); recording {
					p = makeView(p, p.Shape, p.Strides, p.Offset)
				}
			}
			out = gpuConcat(axis, out, p)
		}
		return out
	}
	v := cpuAlloc(numel(s))
	outer := numel(s[:axis])
	inner := numel(s[axis+1:])
	width := s[axis] * inner
	sizes, offsets := make([]int, len(inputs)), make([]int, len(inputs))
	offset := 0
	for i, x := range inputs {
		sizes[i], offsets[i] = x.Shape[axis]*inner, offset
		offset += sizes[i]
	}
	for o := 0; o < outer; o++ {
		for i, x := range inputs {
			size := sizes[i]
			copy(v[o*width+offsets[i]:], x.Data[o*size:(o+1)*size])
		}
	}
	return result(v, s, inputs, func(g []float32) {
		for i, x := range inputs {
			if !x.RequiresGrad {
				continue
			}
			dx := cpuAlloc(outer * sizes[i])
			for o := 0; o < outer; o++ {
				size := sizes[i]
				copy(dx[o*size:(o+1)*size], g[o*width+offsets[i]:])
			}
			x.addGrad(dx)
			cpuRelease(dx)
		}
	})
}
func Slice(a *Tensor, axis, start, end int) *Tensor {
	dispatchBackend("slice", a.Device)
	if axis < 0 {
		axis += len(a.Shape)
	}
	if axis < 0 || axis >= len(a.Shape) || start < 0 || end < start || end > a.Shape[axis] {
		panic("autograd: slice bounds")
	}
	s := append([]int(nil), a.Shape...)
	s[axis] = end - start
	if inferenceDevice(a.Device) {
		inferenceOperands(a)
		offset := a.Offset + start*a.Strides[axis]
		if numel(s) == 0 {
			offset = a.Offset
		}
		return makeView(a, s, a.Strides, offset)
	}
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
	chunk := max(1, count/workers/4)
	jobs := make(chan int, (count-1)/chunk+1)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if chunk == 1 {
				for z := range jobs {
					fn(z)
				}
				return
			}
			for start := range jobs {
				end := start + min(chunk, count-start)
				for z := start; z < end; z++ {
					fn(z)
				}
			}
		}()
	}
	for start := 0; start < count; start += min(chunk, count-start) {
		jobs <- start
	}
	close(jobs)
	wg.Wait()
}
func MatMul(a, b *Tensor) *Tensor {
	if inferenceDevice(a.Device) {
		return inferenceMatMul(a, b)
	}
	useCUDA := dispatchBackend("matmul", a.Device)
	same(a, b)
	a = a.Contiguous()
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
	count := numel(batch)
	aBatch, bBatch := a.Shape[:len(a.Shape)-2], b.Shape[:len(b.Shape)-2]
	aMatrices, bMatrices := numel(aBatch), numel(bBatch)
	flatLinear := count > 1 && aMatrices == count && bMatrices == 1
	transposedB := b.Offset == 0 && lastTwoTranspose(b.Shape, b.Strides, b.Offset) &&
		((!useCUDA && len(b.Data) == b.Numel()) || (useCUDA && b.buf != nil && b.buf.Size() == b.Numel()*4))
	if !transposedB {
		b = b.Contiguous()
	}
	if useCUDA {
		return gpuMatMul(a, b, s, m, n, k, batch, transposedB)
	}
	// Divide the CPU budget between independent batches and each GEMM. In
	// particular, an ordinary rank-two product must not be capped at one core.
	gemmWorkers := max(1, runtime.GOMAXPROCS(0)/max(1, count))
	v := cpuAlloc(numel(s))
	var indices *[2]broadcastIndex
	if flatLinear || count == 1 {
		tensor.SGEMMOp(v, a.Data, b.Data, count*m, n, k, false, transposedB)
	} else {
		if aMatrices != count || bMatrices != count {
			indices = &[2]broadcastIndex{makeBroadcastIndex(batch, aBatch), makeBroadcastIndex(batch, bBatch)}
		}
		parallelBatches(count, func(z int) {
			ai, bi := z*m*k, z*k*n
			if indices != nil {
				ai = indices[0].at(z) * m * k
				bi = indices[1].at(z) * k * n
			}
			tensor.SGEMMOpWorkers(v[z*m*n:(z+1)*m*n], a.Data[ai:ai+m*k], b.Data[bi:bi+k*n], m, n, k, false, transposedB, gemmWorkers)
		})
	}
	return result(v, s, []*Tensor{a, b}, func(g []float32) {
		var da, db []float32
		if a.RequiresGrad {
			da = cpuAlloc(len(a.Data))
			defer cpuRelease(da)
		}
		if b.RequiresGrad {
			db = cpuAlloc(len(b.Data))
			defer cpuRelease(db)
		}
		if flatLinear || count == 1 {
			if a.RequiresGrad {
				// A materialized b is [k,n]; the transposed view holds [n,k].
				tensor.SGEMMOp(da, g, b.Data, count*m, k, n, false, !transposedB)
				a.addGrad(da)
			}
			if b.RequiresGrad {
				tensor.SGEMMOp(db, a.Data, g, k, n, count*m, true, false)
				b.addGrad(db)
			}
			return
		}
		if count > 1 && indices == nil {
			parallelBatches(count, func(z int) {
				ai := z * m * k
				bi := z * k * n
				gg := g[z*m*n : (z+1)*m*n]
				if a.RequiresGrad {
					tensor.SGEMMOpWorkers(da[ai:ai+m*k], gg, b.Data[bi:bi+k*n], m, k, n, false, !transposedB, gemmWorkers)
				}
				if b.RequiresGrad {
					tensor.SGEMMOpWorkers(db[bi:bi+k*n], a.Data[ai:ai+m*k], gg, k, n, m, true, false, gemmWorkers)
				}
			})
			a.addGrad(da)
			b.addGrad(db)
			return
		}
		var tmpA, tmpB []float32
		if a.RequiresGrad {
			tmpA = cpuAlloc(m * k)
			defer cpuRelease(tmpA)
		}
		if b.RequiresGrad {
			tmpB = cpuAlloc(k * n)
			defer cpuRelease(tmpB)
		}
		for z := 0; z < count; z++ {
			ai := indices[0].at(z) * m * k
			bi := indices[1].at(z) * k * n
			gg := g[z*m*n : (z+1)*m*n]
			if a.RequiresGrad {
				tensor.SGEMMOp(tmpA, gg, b.Data[bi:bi+k*n], m, k, n, false, !transposedB)
			}
			if b.RequiresGrad {
				tensor.SGEMMOp(tmpB, a.Data[ai:ai+m*k], gg, k, n, m, true, false)
			}
			for i, v := range tmpA {
				da[ai+i] += v
			}
			for i, v := range tmpB {
				db[bi+i] += v
			}
		}
		a.addGrad(da)
		b.addGrad(db)
	})
}

func matMulTransB(a, w *Tensor) *Tensor {
	if inferenceDevice(a.Device) {
		wt := Transpose(w, 0, 1)
		defer func() {
			if r := recover(); r != nil {
				wt.Close()
				panic(r)
			}
		}()
		out := MatMul(a, wt)
		wt.releaseRequested.Store(true)
		return out
	}
	useCUDA := dispatchBackend("matmul", a.Device)
	same(a, w)
	a, w = a.Contiguous(), w.Contiguous()
	if len(a.Shape) < 2 || len(w.Shape) != 2 {
		panic("autograd: matmul rank")
	}
	k := a.Shape[len(a.Shape)-1]
	if w.Shape[1] != k {
		panic("autograd: matmul inner dimension")
	}
	n := w.Shape[0]
	s := append(append([]int(nil), a.Shape[:len(a.Shape)-1]...), n)
	rows := numel(a.Shape[:len(a.Shape)-1])
	if useCUDA {
		return gpuMatMulTransB(a, w, s, rows, n, k)
	}
	v := cpuAllocUninit(rows * n)
	tensor.SGEMMOp(v, a.Data, w.Data, rows, n, k, false, true)
	return result(v, s, []*Tensor{a, w}, func(g []float32) {
		if a.RequiresGrad {
			da := cpuAllocUninit(rows * k)
			tensor.SGEMMOp(da, g, w.Data, rows, k, n, false, false)
			a.addGrad(da)
			cpuRelease(da)
		}
		if w.RequiresGrad {
			dw := cpuAllocUninit(n * k)
			tensor.SGEMMOp(dw, g, a.Data, n, k, rows, true, false)
			w.addGrad(dw)
			cpuRelease(dw)
		}
	})
}
func Embedding(weight *Tensor, ids []int, shape []int) *Tensor {
	dispatchBackend("embedding", weight.Device)
	return EmbeddingWithPadding(weight, ids, shape, -1)
}

// EmbeddingWithPadding gathers rows and leaves the padding row's gradient zero.
func EmbeddingWithPadding(weight *Tensor, ids []int, shape []int, paddingIdx int) *Tensor {
	useCUDA := dispatchBackend("embedding_padding", weight.Device)
	weight = weight.Contiguous()
	if len(weight.Shape) != 2 || numel(shape) != len(ids) {
		panic("autograd: embedding shape")
	}
	if paddingIdx < -1 || paddingIdx >= weight.Shape[0] {
		panic("autograd: padding index")
	}
	// Host metadata belongs to the caller, not the saved backward state.
	ids = append([]int(nil), ids...)
	shape = append([]int(nil), shape...)
	dim := weight.Shape[1]
	s := append(append([]int(nil), shape...), dim)
	if useCUDA {
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
	useCUDA := dispatchBackend("masked_loss", pred.Device)
	same(pred, target)
	pred, target = pred.Contiguous(), target.Contiguous()
	if len(pred.Shape) != 3 || pred.Shape[2] < 1 || !slices.Equal(pred.Shape, target.Shape) {
		panic("autograd: loss shape")
	}
	if useCUDA {
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
	return resultWithSaved(value, []int{}, []*Tensor{pred}, []*Tensor{target}, func(up []float32) {
		dx := make([]float32, len(g))
		for i := range g {
			dx[i] = g[i] * up[0]
		}
		pred.addGrad(dx)
	})
}
