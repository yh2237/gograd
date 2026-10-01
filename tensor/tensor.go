// Package tensor provides a contiguous float32 GPU tensor with a shape on top
// of the cuda device buffers. It carries no autograd; the nn package builds
// modules and gradients from it.
package tensor

import (
	"fmt"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// Tensor is a contiguous, row-major float32 device tensor.
type Tensor struct {
	buf   *cuda.Buffer
	shape []int
}

// Numel returns the number of elements described by shape.
func Numel(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

// New allocates an uninitialized tensor with the given shape.
func New(shape ...int) (*Tensor, error) {
	buf, err := cuda.Alloc(Numel(shape) * 4)
	if err != nil {
		return nil, err
	}
	return &Tensor{buf: buf, shape: append([]int(nil), shape...)}, nil
}

// Zeros returns a zero-filled tensor.
func Zeros(shape ...int) (*Tensor, error) {
	t, err := New(shape...)
	if err != nil {
		return nil, err
	}
	if err := t.Zero(); err != nil {
		t.Close()
		return nil, err
	}
	return t, nil
}

// FromHost uploads host data with the given shape.
func FromHost(shape []int, data []float32) (*Tensor, error) {
	if Numel(shape) != len(data) {
		return nil, fmt.Errorf("tensor: data length %d does not match shape %v", len(data), shape)
	}
	t, err := New(shape...)
	if err != nil {
		return nil, err
	}
	if err := t.CopyFrom(data); err != nil {
		t.Close()
		return nil, err
	}
	return t, nil
}

// Shape returns a copy of the tensor shape.
func (t *Tensor) Shape() []int { return append([]int(nil), t.shape...) }

// Numel returns the element count.
func (t *Tensor) Numel() int { return Numel(t.shape) }

// Buffer returns the underlying device buffer.
func (t *Tensor) Buffer() *cuda.Buffer { return t.buf }

// ToHost downloads the tensor.
func (t *Tensor) ToHost() ([]float32, error) {
	out := make([]float32, t.Numel())
	if err := t.buf.CopyToHost(bytesOf(out)); err != nil {
		return nil, err
	}
	return out, nil
}

// CopyFrom uploads host data into the tensor.
func (t *Tensor) CopyFrom(data []float32) error {
	if len(data) != t.Numel() {
		return fmt.Errorf("tensor: data length %d does not match %d elements", len(data), t.Numel())
	}
	return t.buf.CopyFromHost(bytesOf(data))
}

// Zero clears the tensor.
func (t *Tensor) Zero() error { return t.buf.Memset(0, t.buf.Size()) }

// Reshape returns a view with a new shape that shares the same buffer.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if Numel(shape) != t.Numel() {
		return nil, fmt.Errorf("tensor: reshape element count mismatch")
	}
	return &Tensor{buf: t.buf, shape: append([]int(nil), shape...)}, nil
}

// Close returns the device memory to the pool.
func (t *Tensor) Close() {
	if t.buf != nil {
		t.buf.Free()
		t.buf = nil
	}
}

func bytesOf(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
