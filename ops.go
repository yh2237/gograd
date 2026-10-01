package gograd

import "math"

// Add returns the elementwise sum of two equal-shaped tensors.
func Add(a, b *Tensor) *Tensor {
	if !sameShape(a.Shape, b.Shape) {
		panic("gograd: Add requires equal shapes")
	}
	out := Zeros(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	out.parents = []*Tensor{a, b}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += out.Grad[i]
			b.Grad[i] += out.Grad[i]
		}
	}
	return out
}

// Sub returns a-b elementwise for equal-shaped tensors.
func Sub(a, b *Tensor) *Tensor {
	if !sameShape(a.Shape, b.Shape) {
		panic("gograd: Sub requires equal shapes")
	}
	out := Zeros(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	out.parents = []*Tensor{a, b}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += out.Grad[i]
			b.Grad[i] -= out.Grad[i]
		}
	}
	return out
}

// Tanh returns the elementwise hyperbolic tangent.
func Tanh(a *Tensor) *Tensor {
	out := Zeros(a.Shape...)
	for i := range out.Data {
		out.Data[i] = math.Tanh(a.Data[i])
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += (1 - out.Data[i]*out.Data[i]) * out.Grad[i]
		}
	}
	return out
}

// Clamp limits each element to [low, high]. The gradient passes through where
// the input was already inside the range, matching torch.clamp.
func Clamp(a *Tensor, low, high float64) *Tensor {
	out := Zeros(a.Shape...)
	for i := range out.Data {
		out.Data[i] = math.Min(math.Max(a.Data[i], low), high)
	}
	source := append([]float64(nil), a.Data...)
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range out.Grad {
			if source[i] >= low && source[i] <= high {
				a.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

// ScaleNum returns a multiplied by a constant scalar.
func ScaleNum(a *Tensor, s float64) *Tensor {
	out := Zeros(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] * s
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += s * out.Grad[i]
		}
	}
	return out
}

// Linear computes x*w^T+b with x [B,T,I], w [O,I], b [O], returning [B,T,O].
func Linear(x, w, b *Tensor) *Tensor {
	if x.Dims() != 3 || w.Dims() != 2 || b.Dims() != 1 {
		panic("gograd: Linear expects x[B,T,I], w[O,I], b[O]")
	}
	batch, time, in := x.Shape[0], x.Shape[1], x.Shape[2]
	outFeatures := w.Shape[0]
	if w.Shape[1] != in || b.Shape[0] != outFeatures {
		panic("gograd: Linear shape mismatch")
	}
	out := NewTensor([]int{batch, time, outFeatures}, make([]float64, batch*time*outFeatures))
	for bt := 0; bt < batch*time; bt++ {
		xi := bt * in
		yi := bt * outFeatures
		for o := 0; o < outFeatures; o++ {
			sum := b.Data[o]
			wRow := o * in
			for i := 0; i < in; i++ {
				sum += x.Data[xi+i] * w.Data[wRow+i]
			}
			out.Data[yi+o] = sum
		}
	}
	out.parents = []*Tensor{x, w, b}
	out.backward = func() {
		for bt := 0; bt < batch*time; bt++ {
			xi := bt * in
			yi := bt * outFeatures
			for o := 0; o < outFeatures; o++ {
				g := out.Grad[yi+o]
				if g == 0 {
					continue
				}
				b.Grad[o] += g
				wRow := o * in
				for i := 0; i < in; i++ {
					w.Grad[wRow+i] += g * x.Data[xi+i]
					x.Grad[xi+i] += g * w.Data[wRow+i]
				}
			}
		}
	}
	return out
}

// Transpose12 swaps the last two dimensions of a 3D tensor.
func Transpose12(a *Tensor) *Tensor {
	if a.Dims() != 3 {
		panic("gograd: Transpose12 expects a 3D tensor")
	}
	d0, d1, d2 := a.Shape[0], a.Shape[1], a.Shape[2]
	out := NewTensor([]int{d0, d2, d1}, make([]float64, d0*d1*d2))
	for i := 0; i < d0; i++ {
		for j := 0; j < d1; j++ {
			for k := 0; k < d2; k++ {
				out.Data[i*d2*d1+k*d1+j] = a.Data[i*d1*d2+j*d2+k]
			}
		}
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := 0; i < d0; i++ {
			for j := 0; j < d1; j++ {
				for k := 0; k < d2; k++ {
					a.Grad[i*d1*d2+j*d2+k] += out.Grad[i*d2*d1+k*d1+j]
				}
			}
		}
	}
	return out
}

// Reshape reinterprets the element order with a new shape. The data is copied
// so that graph nodes never alias.
func Reshape(a *Tensor, shape []int) *Tensor {
	if Numel(shape) != a.Numel() {
		panic("gograd: Reshape element count mismatch")
	}
	out := NewTensor(shape, append([]float64(nil), a.Data...))
	out.parents = []*Tensor{a}
	out.backward = func() {
		for i := range out.Grad {
			a.Grad[i] += out.Grad[i]
		}
	}
	return out
}

// SqueezeLast removes a trailing dimension of size one.
func SqueezeLast(a *Tensor) *Tensor {
	if a.Dims() < 2 || a.Shape[len(a.Shape)-1] != 1 {
		panic("gograd: SqueezeLast expects a trailing size-one dimension")
	}
	return Reshape(a, append([]int(nil), a.Shape[:len(a.Shape)-1]...))
}

// Diff returns a[:,1:]-a[:,:-1] for a [B,T] tensor, producing [B,T-1].
func Diff(a *Tensor) *Tensor {
	if a.Dims() != 2 {
		panic("gograd: Diff expects a 2D tensor")
	}
	batch, time := a.Shape[0], a.Shape[1]
	if time < 2 {
		return Zeros(batch, 0)
	}
	out := NewTensor([]int{batch, time - 1}, make([]float64, batch*(time-1)))
	for row := 0; row < batch; row++ {
		base := row * time
		for t := 0; t < time-1; t++ {
			out.Data[row*(time-1)+t] = a.Data[base+t+1] - a.Data[base+t]
		}
	}
	out.parents = []*Tensor{a}
	out.backward = func() {
		for row := 0; row < batch; row++ {
			base := row * time
			obase := row * (time - 1)
			for t := 0; t < time-1; t++ {
				g := out.Grad[obase+t]
				a.Grad[base+t+1] += g
				a.Grad[base+t] -= g
			}
		}
	}
	return out
}

// GatherMask selects the elements flagged by mask in row-major order.
func GatherMask(a *Tensor, mask []bool) *Tensor {
	if len(mask) != a.Numel() {
		panic("gograd: GatherMask mask length mismatch")
	}
	count := 0
	for _, keep := range mask {
		if keep {
			count++
		}
	}
	data := make([]float64, 0, count)
	for i, keep := range mask {
		if keep {
			data = append(data, a.Data[i])
		}
	}
	out := NewTensor([]int{count}, data)
	out.parents = []*Tensor{a}
	out.backward = func() {
		cursor := 0
		for i, keep := range mask {
			if keep {
				a.Grad[i] += out.Grad[cursor]
				cursor++
			}
		}
	}
	return out
}
