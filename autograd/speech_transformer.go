package autograd

import (
	"math"
)

// RotaryHalf applies half-rotation RoPE to [batch,heads,sequence,headDim].
// ModernBERT uses this layout and a different theta for global/local layers.
func RotaryHalf(x *Tensor, theta float64) *Tensor {
	dispatchBackend("rotary_half", x.Device)
	if len(x.Shape) != 4 || x.Shape[3]%2 != 0 || x.Shape[3] == 0 || theta <= 0 {
		panic("autograd: invalid RotaryHalf input")
	}
	seq, half := x.Shape[2], x.Shape[3]/2
	co, si := make([]float32, seq*half), make([]float32, seq*half)
	for pos := 0; pos < seq; pos++ {
		for d := 0; d < half; d++ {
			angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(2*half))
			co[pos*half+d] = float32(math.Cos(angle))
			si[pos*half+d] = float32(math.Sin(angle))
		}
	}
	c, err := New(co, []int{1, 1, seq, half}, x.Device, false)
	if err != nil {
		panic(err)
	}
	s, err := New(si, []int{1, 1, seq, half}, x.Device, false)
	if err != nil {
		c.Close()
		panic(err)
	}
	c.ephemeral, s.ephemeral = true, true
	c.execution, s.execution = x.execution, x.execution
	first, second := Slice(x, 3, 0, half), Slice(x, 3, half, 2*half)
	return Concat(3, Sub(Mul(first, c), Mul(second, s)), Add(Mul(second, c), Mul(first, s)))
}

// GeGLU applies the ModernBERT projection order: GELU(first half) * second.
func GeGLU(x *Tensor) *Tensor {
	dispatchBackend("geglu", x.Device)
	if len(x.Shape) == 0 || x.Shape[len(x.Shape)-1]%2 != 0 {
		panic("autograd: invalid GeGLU shape")
	}
	axis := len(x.Shape) - 1
	half := x.Shape[axis] / 2
	return Mul(GELU(Slice(x, axis, 0, half), false), Slice(x, axis, half, 2*half))
}

// SwiGLUProjection applies SiLU(x*w1^T) * (x*w3^T), then w2^T.
// Weights have the PyTorch [out,in] layout.
func SwiGLUProjection(x, w1, w2, w3 *Tensor) *Tensor {
	dispatchBackend("swiglu_projection", x.Device)
	if len(x.Shape) < 2 || len(w1.Shape) != 2 || len(w2.Shape) != 2 || len(w3.Shape) != 2 || w1.Shape[0] != w3.Shape[0] || w1.Shape[1] != x.Shape[len(x.Shape)-1] || w3.Shape[1] != w1.Shape[1] || w2.Shape[1] != w1.Shape[0] || w2.Shape[0] != w1.Shape[1] {
		panic("autograd: invalid SwiGLU shape")
	}
	a := matMulTransB(x, w1)
	b := matMulTransB(x, w3)
	silu := Mul(a, Sigmoid(a))
	return matMulTransB(Mul(silu, b), w2)
}

// KeyPaddingAttention converts a boolean key mask to an additive attention
// mask. A negative radius selects full attention; otherwise keys outside the
// inclusive query-centered window are masked. q/k/v are [batch,heads,seq,dim].
func KeyPaddingAttention(q, k, v *Tensor, valid []bool, radius int) *Tensor {
	dispatchBackend("key_padding_attention", q.Device)
	if len(q.Shape) != 4 || len(k.Shape) != 4 || len(v.Shape) != 4 || q.Shape[0] != k.Shape[0] || k.Shape[0] != v.Shape[0] || q.Shape[1] != k.Shape[1] || k.Shape[1] != v.Shape[1] || k.Shape[2] != v.Shape[2] || q.Shape[3] != k.Shape[3] || k.Shape[3] != v.Shape[3] || len(valid) != q.Shape[0]*k.Shape[2] {
		panic("autograd: invalid attention shape")
	}
	batch, queries, keys := q.Shape[0], q.Shape[2], k.Shape[2]
	mask := make([]float32, batch*queries*keys)
	for b := 0; b < batch; b++ {
		for i := 0; i < queries; i++ {
			count := 0
			for j := 0; j < keys; j++ {
				if valid[b*keys+j] && (radius < 0 || (j >= i-radius && j <= i+radius)) {
					count++
				} else {
					mask[(b*queries+i)*keys+j] = -1e9
				}
			}
			if count == 0 {
				panic("autograd: fully masked attention query")
			}
		}
	}
	m, err := New(mask, []int{batch, 1, queries, keys}, q.Device, false)
	if err != nil {
		panic(err)
	}
	m.ephemeral = true
	m.execution = q.execution
	return ScaledDotProductAttention(q, k, v, m)
}
