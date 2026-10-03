package autograd

import "math"

// Conv1d uses [batch,time,channels] and [out,in,kernel] weights.
func Conv1d(x, w, b *Tensor, dilation int) *Tensor {
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
	same(x, w)
	same(x, b)
	s := x.Shape
	if len(s) != 3 || groups < 1 || s[2]%groups != 0 || len(w.Data) != s[2] || len(b.Data) != s[2] {
		panic("autograd: groupnorm shape")
	}
	bs, t, c := s[0], s[1], s[2]
	cg := c / groups
	size := float32(t * cg)
	v := make([]float32, len(x.Data))
	norm := make([]float32, len(x.Data))
	inv := make([]float32, bs*groups)
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
		dx := make([]float32, len(g))
		dw := make([]float32, c)
		db := make([]float32, c)
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
	})
}

// LayerNorm normalizes over the last dimension.
func LayerNorm(x, w, b *Tensor, eps float32) *Tensor {
	if len(x.Shape) < 1 {
		panic("autograd: layernorm rank")
	}
	s := x.Shape
	c := s[len(s)-1]
	flat := Reshape(x, len(x.Data)/c, 1, c)
	y := GroupNorm(flat, w, b, 1, eps)
	return Reshape(y, s...)
}

type Parameter struct {
	Name  string
	Value *Tensor
}

func ClipGradNorm(params []Parameter, maxNorm float32) float32 {
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
	StepCount                          int
}

func NewAdamW(params []Parameter, lr, decay float32) *AdamW {
	o := &AdamW{Params: params, LR: lr, WeightDecay: decay, Beta1: .9, Beta2: .999, Eps: 1e-8}
	for _, p := range params {
		o.M = append(o.M, make([]float32, len(p.Value.Data)))
		o.V = append(o.V, make([]float32, len(p.Value.Data)))
	}
	return o
}
func (o *AdamW) Step() {
	o.StepCount++
	bc1 := float32(1 - math.Pow(float64(o.Beta1), float64(o.StepCount)))
	bc2 := float32(1 - math.Pow(float64(o.Beta2), float64(o.StepCount)))
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
