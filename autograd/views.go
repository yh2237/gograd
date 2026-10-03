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
func makeView(a *Tensor, shape, stride []int, offset int) *Tensor {
	if !a.IsContiguous() {
		panic("autograd: view parent must be contiguous")
	}
	v := &Tensor{Data: a.Data, Shape: append([]int(nil), shape...), Strides: append([]int(nil), stride...), Offset: offset, DType: Float32, Device: a.Device, buf: a.buf, RequiresGrad: a.RequiresGrad && recording, parents: []*Tensor{a}}
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
			for i, z := range g {
				dx[storageIndex(i, v.Shape, v.Strides, v.Offset)] += z
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
	if t.IsContiguous() {
		return t
	}
	n := t.Numel()
	if t.Device == tensor.CPU {
		v := cpuAlloc(n)
		for i := range v {
			v[i] = t.Data[storageIndex(i, t.Shape, t.Strides, t.Offset)]
		}
		return result(v, append([]int(nil), t.Shape...), []*Tensor{t}, func(g []float32) { t.addGrad(g) })
	}
	out := mustAlloc(n)
	gpuViewGather(t.buf, out, t.Shape, t.Strides, t.Offset)
	return resultGPU(out, append([]int(nil), t.Shape...), []*Tensor{t}, func(g *cuda.Buffer) {
		if dst := t.ensureGradGPU(); dst != nil {
			addDevice(dst, g, n)
		}
	})
}

// Expand broadcasts singleton dimensions without copying storage.
func Expand(a *Tensor, shape ...int) *Tensor {
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
