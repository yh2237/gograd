// Package gograd is a small reverse-mode autograd library in Go. The first
// supported model is a dilated residual TCN used for frame-level intonation,
// and the operation set is limited to what that model and its loss require.
package gograd

import "fmt"

// Tensor is a contiguous, row-major, float64 tensor with an eager reverse-mode
// graph. Only the operations implemented in this package are differentiable.
type Tensor struct {
	Data  []float64
	Grad  []float64
	Shape []int

	parents  []*Tensor
	backward func()
}

// Numel returns the number of elements described by shape.
func Numel(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

// NewTensor wraps data with the given shape. data is used directly and is not
// copied.
func NewTensor(shape []int, data []float64) *Tensor {
	n := Numel(shape)
	if len(data) != n {
		panic(fmt.Sprintf("gograd: data length %d does not match shape %v (%d)", len(data), shape, n))
	}
	return &Tensor{
		Data:  data,
		Grad:  make([]float64, n),
		Shape: append([]int(nil), shape...),
	}
}

// Zeros returns a zero-filled tensor with the given shape.
func Zeros(shape ...int) *Tensor {
	return NewTensor(shape, make([]float64, Numel(shape)))
}

// Numel reports the number of elements in the tensor.
func (t *Tensor) Numel() int { return len(t.Data) }

// Dims reports the number of dimensions.
func (t *Tensor) Dims() int { return len(t.Shape) }

// Size reports the extent of the given dimension.
func (t *Tensor) Size(dim int) int { return t.Shape[dim] }

// ZeroGrad clears the accumulated gradient.
func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// Backward propagates gradients from this scalar tensor to its leaves. The
// seed gradient is 1, matching the implicit derivative of the loss.
func (t *Tensor) Backward() {
	if t.Numel() != 1 {
		panic("gograd: Backward requires a scalar tensor")
	}
	order := make([]*Tensor, 0, 32)
	visited := make(map[*Tensor]bool)
	var visit func(n *Tensor)
	visit = func(n *Tensor) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, parent := range n.parents {
			visit(parent)
		}
		order = append(order, n)
	}
	visit(t)
	for _, node := range order {
		for i := range node.Grad {
			node.Grad[i] = 0
		}
	}
	t.Grad[0] = 1
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
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

// FlattenBool converts a row-major [][]bool mask into a flat mask.
func FlattenBool(mask [][]bool) []bool {
	total := 0
	for _, row := range mask {
		total += len(row)
	}
	flat := make([]bool, 0, total)
	for _, row := range mask {
		flat = append(flat, row...)
	}
	return flat
}
