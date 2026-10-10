package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func storageIndex(flat int, shape, stride []int, offset int) int {
	j := offset
	for axis := len(shape) - 1; axis >= 0; axis-- {
		if shape[axis] == 0 {
			return j
		}
		j += (flat % shape[axis]) * stride[axis]
		flat /= shape[axis]
	}
	return j
}

// A common attention view swaps the last two axes of contiguous matrices.
// Detect it once and copy tiles without per-element N-D division/modulo.
func lastTwoTranspose(shape, stride []int, offset int) bool {
	n := len(shape)
	if n < 2 || offset != 0 || stride[n-2] != 1 || stride[n-1] != shape[n-2] {
		return false
	}
	expected := shape[n-2] * shape[n-1]
	for i := n - 3; i >= 0; i-- {
		if stride[i] != expected {
			return false
		}
		expected *= shape[i]
	}
	return true
}

func transposeViewCopy(dst, src []float32, shape []int, backward bool) {
	n := len(shape)
	rows, cols := shape[n-2], shape[n-1]
	outer := numel(shape[:n-2])
	tiles := (rows + 31) / 32
	parallelFor(outer*tiles*cols, func(start, end int) {
		// Split on (outer,row-tile,column), keeping each source span contiguous.
		for task := start; task < end; task++ {
			o := task / (tiles * cols)
			rem := task % (tiles * cols)
			r0, c := rem/cols*32, rem%cols
			r1 := min(rows, r0+32)
			base := o * rows * cols
			for r := r0; r < r1; r++ {
				if backward {
					dst[base+c*rows+r] += src[base+r*cols+c]
				} else {
					dst[base+r*cols+c] = src[base+c*rows+r]
				}
			}
		}
	})
}
func makeView(a *Tensor, shape, stride []int, offset int) *Tensor {
	a.assertOpen()
	if !a.IsContiguous() {
		panic("autograd: view parent must be contiguous")
	}
	context, req := graphRecording([]*Tensor{a})
	v := &Tensor{Data: a.Data, storage: a.storage.acquire(), outputVersion: a.storage.version.Load(), Shape: append([]int(nil), shape...), Strides: append([]int(nil), stride...), Offset: offset, DType: Float32, Device: a.Device, buf: a.buf, RequiresGrad: req, execution: context, intermediate: true}
	v.attachParents([]*Tensor{a})
	v.shareBF16(a)
	if !req {
		return v
	}
	identity := offset == 0
	want := strides(shape)
	for i := range want {
		if stride[i] != want[i] {
			identity = false
			break
		}
	}
	if a.Device == tensor.CPU {
		v.backward = func(g []float32) {
			if !a.RequiresGrad {
				return
			}
			if identity {
				a.addGrad(g)
				return
			}
			dx := cpuAlloc(a.Numel())
			if lastTwoTranspose(v.Shape, v.Strides, v.Offset) {
				transposeViewCopy(dx, g, v.Shape, true)
			} else {
				for i, z := range g {
					dx[storageIndex(i, v.Shape, v.Strides, v.Offset)] += z
				}
			}
			a.addGrad(dx)
			cpuRelease(dx)
		}
	} else {
		v.backwardGPU = func(g *cuda.Buffer) {
			if dst := a.ensureGradGPU(); dst != nil {
				if identity {
					addDevice(dst, g, v.Numel())
				} else {
					gpuViewScatter(g, dst, v.Shape, v.Strides, v.Offset)
				}
			}
		}
	}
	return v
}

// Contiguous materializes a view in row-major order. A contiguous tensor is
// returned unchanged. Backward maps the copy's logical gradient to its view.
func (t *Tensor) Contiguous() *Tensor {
	t.assertOpen()
	useCUDA := dispatchBackend("contiguous", t.Device)
	if t.IsContiguous() {
		return t
	}
	n := t.Numel()
	if !useCUDA {
		v := cpuAlloc(n)
		if lastTwoTranspose(t.Shape, t.Strides, t.Offset) {
			transposeViewCopy(v, t.Data, t.Shape, false)
		} else {
			for i := range v {
				v[i] = t.Data[storageIndex(i, t.Shape, t.Strides, t.Offset)]
			}
		}
		return result(v, append([]int(nil), t.Shape...), []*Tensor{t}, func(g []float32) { t.addGrad(g) })
	}
	out := mustAlloc(n)
	gpuViewGather(t.buf, out, t.Shape, t.Strides, t.Offset)
	res := resultGPU(out, append([]int(nil), t.Shape...), []*Tensor{t}, func(g *cuda.Buffer) {
		if dst := t.ensureGradGPU(); dst != nil {
			addDevice(dst, g, n)
		}
	})
	if executionOptions(t).BF16Autocast && t.currentBF16() != nil {
		shadow := allocBF16(n)
		gpuViewGatherBF16(t.currentBF16(), shadow, t.Shape, t.Strides, t.Offset)
		res.setBF16(shadow)
	}
	return res
}

// Expand broadcasts singleton dimensions without copying storage.
func Expand(a *Tensor, shape ...int) *Tensor {
	dispatchBackend("expand", a.Device)
	if len(shape) < len(a.Shape) || len(shape) > 6 {
		panic("autograd: expand rank")
	}
	a = a.Contiguous()
	str := make([]int, len(shape))
	pad := len(shape) - len(a.Shape)
	for i, d := range shape {
		old := 1
		if i >= pad {
			old = a.Shape[i-pad]
		}
		if old != d && old != 1 {
			panic("autograd: expand shape")
		}
		if i >= pad && old != 1 {
			str[i] = a.Strides[i-pad]
		}
	}
	return makeView(a, shape, str, 0)
}
