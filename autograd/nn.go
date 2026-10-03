package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"github.com/yh2237/gograd/tensor"
	"math"
	"sync"
	"unsafe"
)

var lastNormGPU *cuda.Buffer

// GradientNormToHost is a diagnostic readback for tests and profiling.
func GradientNormToHost() float32 {
	if lastNormGPU == nil {
		return 0
	}
	v := []float32{0}
	if e := lastNormGPU.CopyToHost(floatBytes(v)); e != nil {
		panic(e)
	}
	return v[0]
}

// Conv1d uses [batch,time,channels] and [out,in,kernel] weights.
func Conv1d(x, w, b *Tensor, dilation int) *Tensor {
	useCUDA := dispatchBackend("conv1d", x.Device)
	x, w, b = x.Contiguous(), w.Contiguous(), b.Contiguous()
	if useCUDA {
		return Conv1dGEMM(x, w, b, dilation)
	}
	same(x, w)
	same(x, b)
	s := x.Shape
	if len(s) != 3 || len(w.Shape) != 3 || w.Shape[1] != s[2] || len(b.Data) != w.Shape[0] || w.Shape[2]%2 != 1 || dilation < 1 {
		panic("autograd: conv1d shape")
	}
	bs, t, ci, co, k := s[0], s[1], s[2], w.Shape[0], w.Shape[2]
	v := make([]float32, bs*t*co)
	pad := k / 2 * dilation
	for n := 0; n < bs; n++ {
		for at := 0; at < t; at++ {
			for o := 0; o < co; o++ {
				z := b.Data[o]
				for i := 0; i < ci; i++ {
					for j := 0; j < k; j++ {
						src := at + (j-k/2)*dilation
						if src >= 0 && src < t {
							z += x.Data[(n*t+src)*ci+i] * w.Data[(o*ci+i)*k+j]
						}
					}
				}
				v[(n*t+at)*co+o] = z
			}
		}
	}
	_ = pad
	return result(v, []int{bs, t, co}, []*Tensor{x, w, b}, func(g []float32) {
		dx := make([]float32, len(x.Data))
		dw := make([]float32, len(w.Data))
		db := make([]float32, len(b.Data))
		for n := 0; n < bs; n++ {
			for at := 0; at < t; at++ {
				for o := 0; o < co; o++ {
					z := g[(n*t+at)*co+o]
					db[o] += z
					for i := 0; i < ci; i++ {
						for j := 0; j < k; j++ {
							src := at + (j-k/2)*dilation
							if src >= 0 && src < t {
								xi := (n*t+src)*ci + i
								wi := (o*ci+i)*k + j
								dx[xi] += z * w.Data[wi]
								dw[wi] += z * x.Data[xi]
							}
						}
					}
				}
			}
		}
		x.addGrad(dx)
		w.addGrad(dw)
		b.addGrad(db)
	})
}

// GroupNorm normalizes each sample across its time and channels per group.
func GroupNorm(x, w, b *Tensor, groups int, eps float32) *Tensor {
	useCUDA := dispatchBackend("group_norm", x.Device)
	same(x, w)
	same(x, b)
	x, w, b = x.Contiguous(), w.Contiguous(), b.Contiguous()
	s := x.Shape
	if len(s) != 3 || groups < 1 || s[2]%groups != 0 || w.Numel() != s[2] || b.Numel() != s[2] {
		panic("autograd: groupnorm shape")
	}
	if useCUDA {
		return gpuGroupNorm(x, w, b, groups, eps)
	}
	bs, t, c := s[0], s[1], s[2]
	cg := c / groups
	size := float32(t * cg)
	v := cpuAlloc(len(x.Data))
	norm := cpuAlloc(len(x.Data))
	inv := cpuAlloc(bs * groups)
	for n := 0; n < bs; n++ {
		for gr := 0; gr < groups; gr++ {
			var mean, variance float32
			for at := 0; at < t; at++ {
				for ch := gr * cg; ch < (gr+1)*cg; ch++ {
					mean += x.Data[(n*t+at)*c+ch]
				}
			}
			mean /= size
			for at := 0; at < t; at++ {
				for ch := gr * cg; ch < (gr+1)*cg; ch++ {
					d := x.Data[(n*t+at)*c+ch] - mean
					variance += d * d
				}
			}
			variance /= size
			iv := float32(1 / math.Sqrt(float64(variance+eps)))
			inv[n*groups+gr] = iv
			for at := 0; at < t; at++ {
				for ch := gr * cg; ch < (gr+1)*cg; ch++ {
					idx := (n*t+at)*c + ch
					norm[idx] = (x.Data[idx] - mean) * iv
					v[idx] = norm[idx]*w.Data[ch] + b.Data[ch]
				}
			}
		}
	}
	return result(v, s, []*Tensor{x, w, b}, func(g []float32) {
		defer cpuRelease(norm)
		defer cpuRelease(inv)
		dx := cpuAlloc(len(g))
		dw := cpuAlloc(c)
		db := cpuAlloc(c)
		for n := 0; n < bs; n++ {
			for gr := 0; gr < groups; gr++ {
				var sg, sgn float32
				for at := 0; at < t; at++ {
					for ch := gr * cg; ch < (gr+1)*cg; ch++ {
						idx := (n*t+at)*c + ch
						z := g[idx] * w.Data[ch]
						sg += z
						sgn += z * norm[idx]
						dw[ch] += g[idx] * norm[idx]
						db[ch] += g[idx]
					}
				}
				for at := 0; at < t; at++ {
					for ch := gr * cg; ch < (gr+1)*cg; ch++ {
						idx := (n*t+at)*c + ch
						dx[idx] = inv[n*groups+gr] * (size*g[idx]*w.Data[ch] - sg - norm[idx]*sgn) / size
					}
				}
			}
		}
		x.addGrad(dx)
		w.addGrad(dw)
		b.addGrad(db)
		cpuRelease(dx)
		cpuRelease(dw)
		cpuRelease(db)
	})
}

// LayerNorm normalizes over the last dimension.
func LayerNorm(x, w, b *Tensor, eps float32) *Tensor {
	useCUDA := dispatchBackend("layer_norm", x.Device)
	if len(x.Shape) < 1 {
		panic("autograd: layernorm rank")
	}
	same(x, w)
	same(x, b)
	x, w, b = x.Contiguous(), w.Contiguous(), b.Contiguous()
	c := x.Shape[len(x.Shape)-1]
	if c < 1 || len(w.Shape) != 1 || len(b.Shape) != 1 || w.Numel() != c || b.Numel() != c {
		panic("autograd: layernorm shape")
	}
	if useCUDA {
		return gpuLayerNorm(x, w, b, eps)
	}
	rows := x.Numel() / c
	out, means, invs := cpuAlloc(x.Numel()), cpuAlloc(rows), cpuAlloc(rows)
	parallelFor(rows, func(start, end int) {
		for row := start; row < end; row++ {
			base := row * c
			var mean, varsum float32
			for j := 0; j < c; j++ {
				mean += x.Data[base+j]
			}
			mean /= float32(c)
			for j := 0; j < c; j++ {
				z := x.Data[base+j] - mean
				varsum += z * z
			}
			inv := float32(1 / math.Sqrt(float64(varsum/float32(c)+eps)))
			means[row], invs[row] = mean, inv
			for j := 0; j < c; j++ {
				out[base+j] = (x.Data[base+j]-mean)*inv*w.Data[j] + b.Data[j]
			}
		}
	})
	return result(out, x.Shape, []*Tensor{x, w, b}, func(g []float32) {
		defer cpuRelease(means)
		defer cpuRelease(invs)
		dx, dw, db := cpuAlloc(x.Numel()), cpuAlloc(c), cpuAlloc(c)
		var mu sync.Mutex
		parallelFor(rows, func(start, end int) {
			localW, localB := make([]float32, c), make([]float32, c)
			for row := start; row < end; row++ {
				base := row * c
				mean, inv := means[row], invs[row]
				var sg, sgn float32
				for j := 0; j < c; j++ {
					norm := (x.Data[base+j] - mean) * inv
					z := g[base+j] * w.Data[j]
					sg += z
					sgn += z * norm
					localW[j] += g[base+j] * norm
					localB[j] += g[base+j]
				}
				sg /= float32(c)
				sgn /= float32(c)
				for j := 0; j < c; j++ {
					norm := (x.Data[base+j] - mean) * inv
					dx[base+j] = inv * (g[base+j]*w.Data[j] - sg - norm*sgn)
				}
			}
			mu.Lock()
			for j := 0; j < c; j++ {
				dw[j] += localW[j]
				db[j] += localB[j]
			}
			mu.Unlock()
		})
		x.addGrad(dx)
		w.addGrad(dw)
		b.addGrad(db)
		cpuRelease(dx)
		cpuRelease(dw)
		cpuRelease(db)
	})
}

type Parameter struct {
	Name  string
	Value *Tensor
}

func ClipGradNorm(params []Parameter, maxNorm float32) float32 {
	if len(params) > 0 && params[0].Value.Device == tensor.CUDA {
		clipGPU(params, maxNorm)
		return float32(math.NaN())
	}
	var sum float64
	for _, p := range params {
		for _, g := range p.Value.Grad {
			sum += float64(g) * float64(g)
		}
	}
	norm := float32(math.Sqrt(sum))
	if norm > maxNorm && maxNorm > 0 {
		scale := maxNorm / (norm + 1e-6)
		for _, p := range params {
			for i := range p.Value.Grad {
				p.Value.Grad[i] *= scale
			}
		}
	}
	return norm
}

type AdamW struct {
	Params                             []Parameter
	LR, WeightDecay, Beta1, Beta2, Eps float32
	M, V                               [][]float32
	mGPU, vGPU                         []*cuda.Buffer
	unityGPU                           *cuda.Buffer
	graphStep                          *cuda.Buffer
	StepCount                          int
}

// PrepareGraph initializes a device step counter before capturing AdamW.
// Captured replays advance this counter and recompute bias correction.
func (o *AdamW) PrepareGraph() error {
	if len(o.Params) == 0 || o.Params[0].Value.Device != tensor.CUDA {
		return cuda.ErrUnavailable
	}
	if o.graphStep == nil {
		o.graphStep = mustAlloc(1)
	}
	step := int32(o.StepCount)
	return o.graphStep.CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&step)), 4))
}
func (o *AdamW) GraphStepCount() (int, error) {
	if o.graphStep == nil {
		return 0, cuda.ErrUnavailable
	}
	var step int32
	if e := o.graphStep.CopyToHost(unsafe.Slice((*byte)(unsafe.Pointer(&step)), 4)); e != nil {
		return 0, e
	}
	return int(step), nil
}

// SyncGraphStepCount restores the host counter before switching from graph
// replay back to eager AdamW updates. It reads one scalar from the device.
func (o *AdamW) SyncGraphStepCount() error {
	step, e := o.GraphStepCount()
	if e != nil {
		return e
	}
	o.StepCount = step
	return nil
}

func NewAdamW(params []Parameter, lr, decay float32) *AdamW {
	o := &AdamW{Params: params, LR: lr, WeightDecay: decay, Beta1: .9, Beta2: .999, Eps: 1e-8}
	if len(params) > 0 && params[0].Value.Device == tensor.CUDA {
		o.unityGPU = mustAlloc(1)
		setOne(o.unityGPU)
	}
	for _, p := range params {
		if p.Value.Device == tensor.CUDA {
			m, v := mustAlloc(p.Value.Numel()), mustAlloc(p.Value.Numel())
			m.Memset(0, m.Size())
			v.Memset(0, v.Size())
			o.mGPU = append(o.mGPU, m)
			o.vGPU = append(o.vGPU, v)
		} else {
			o.M = append(o.M, make([]float32, p.Value.Numel()))
			o.V = append(o.V, make([]float32, p.Value.Numel()))
		}
	}
	return o
}
func (o *AdamW) Step() {
	o.StepCount++
	if cuda.InCapture() {
		if o.graphStep == nil {
			panic("autograd: call AdamW.PrepareGraph before capture")
		}
		step := ptr(o.graphStep)
		launch("graph_adam_inc", 1, unsafe.Pointer(&step))
		for i, p := range o.Params {
			if p.Value.gradBuf == nil {
				continue
			}
			pp, gp, mp, vp := ptr(p.Value.buf), ptr(p.Value.gradBuf), ptr(o.mGPU[i]), ptr(o.vGPU[i])
			n := int32(p.Value.Numel())
			launch("graph_adamw", int(n), unsafe.Pointer(&pp), unsafe.Pointer(&gp), unsafe.Pointer(&mp), unsafe.Pointer(&vp), unsafe.Pointer(&step), unsafe.Pointer(&n), unsafe.Pointer(&o.Beta1), unsafe.Pointer(&o.Beta2), unsafe.Pointer(&o.LR), unsafe.Pointer(&o.WeightDecay), unsafe.Pointer(&o.Eps))
		}
		return
	}
	bc1 := float32(1 - math.Pow(float64(o.Beta1), float64(o.StepCount)))
	bc2 := float32(1 - math.Pow(float64(o.Beta2), float64(o.StepCount)))
	if len(o.Params) > 0 && o.Params[0].Value.Device == tensor.CUDA {
		for n, p := range o.Params {
			if p.Value.gradBuf == nil {
				continue
			}
			if e := kernels.AdamWUpdate(p.Value.buf, p.Value.gradBuf, o.mGPU[n], o.vGPU[n], o.unityGPU, p.Value.Numel(), o.Beta1, o.Beta2, o.LR/bc1, bc2, o.LR*o.WeightDecay, o.Eps, math.MaxFloat32); e != nil {
				panic(e)
			}
			p.Value.invalidateBF16()
		}
		return
	}
	for n, p := range o.Params {
		for i, g := range p.Value.Grad {
			o.M[n][i] = o.Beta1*o.M[n][i] + (1-o.Beta1)*g
			o.V[n][i] = o.Beta2*o.V[n][i] + (1-o.Beta2)*g*g
			p.Value.Data[i] *= 1 - o.LR*o.WeightDecay
			p.Value.Data[i] -= o.LR * (o.M[n][i] / bc1) / (float32(math.Sqrt(float64(o.V[n][i]/bc2))) + o.Eps)
		}
	}
}
func (o *AdamW) ZeroGrad() {
	for _, p := range o.Params {
		p.Value.ZeroGrad()
	}
}

type OneCycle struct {
	MaxLR, PctStart, DivFactor, FinalDivFactor float64
	Total, StepCount                           int
}

func NewOneCycle(maxLR float64, total int, pctStart float64) *OneCycle {
	return &OneCycle{MaxLR: maxLR, Total: total, PctStart: pctStart, DivFactor: 25, FinalDivFactor: 1e4}
}
func (c *OneCycle) LR() float64 {
	initial := c.MaxLR / c.DivFactor
	final := initial / c.FinalDivFactor
	step := float64(c.StepCount)
	warm := c.PctStart*float64(c.Total) - 1
	var start, end, p float64
	if step <= warm {
		start, end = initial, c.MaxLR
		p = step / warm
	} else {
		start, end = c.MaxLR, final
		p = (step - warm) / (float64(c.Total-1) - warm)
	}
	return end + (start-end)*(1+math.Cos(math.Pi*p))/2
}
func (c *OneCycle) Step(o *AdamW) { c.StepCount++; o.LR = float32(c.LR()) }
