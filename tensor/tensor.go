// Package tensor provides contiguous float32 tensors on CPU or CUDA.
package tensor

import (
	"fmt"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// Tensor is a contiguous, row-major float32 device tensor.
type Tensor struct {
	buf    *cuda.Buffer
	data   []float32
	device Device
	shape  []int
}

// Device selects storage at construction time.
type Device string

const (
	CPU  Device = "cpu"
	CUDA Device = "cuda"
)

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
	return NewOn(CUDA, shape...)
}

// NewOn allocates a tensor on the chosen device.
func NewOn(device Device, shape ...int) (*Tensor, error) {
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("tensor: negative dimension")
		}
	}
	if device == CPU {
		return &Tensor{data: make([]float32, Numel(shape)), device: CPU, shape: append([]int(nil), shape...)}, nil
	}
	if device != CUDA {
		return nil, fmt.Errorf("tensor: unknown device %q", device)
	}
	buf, err := cuda.Alloc(Numel(shape) * 4)
	if err != nil {
		return nil, err
	}
	return &Tensor{buf: buf, device: CUDA, shape: append([]int(nil), shape...)}, nil
}

// Zeros returns a zero-filled tensor.
func Zeros(shape ...int) (*Tensor, error) {
	return ZerosOn(CUDA, shape...)
}

// ZerosOn allocates a zero-filled tensor on device.
func ZerosOn(device Device, shape ...int) (*Tensor, error) {
	t, err := NewOn(device, shape...)
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
	return FromHostOn(CUDA, shape, data)
}

// FromHostOn copies data to the chosen device.
func FromHostOn(device Device, shape []int, data []float32) (*Tensor, error) {
	if Numel(shape) != len(data) {
		return nil, fmt.Errorf("tensor: data length %d does not match shape %v", len(data), shape)
	}
	t, err := NewOn(device, shape...)
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

// Device returns the storage device.
func (t *Tensor) Device() Device { return t.device }

// Data returns the CPU backing slice, or nil for CUDA tensors.
func (t *Tensor) Data() []float32 { return t.data }

// ToHost downloads the tensor.
func (t *Tensor) ToHost() ([]float32, error) {
	if t.device == CPU {
		return append([]float32(nil), t.data...), nil
	}
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
	if t.device == CPU {
		copy(t.data, data)
		return nil
	}
	return t.buf.CopyFromHost(bytesOf(data))
}

// Zero clears the tensor.
func (t *Tensor) Zero() error {
	if t.device == CPU {
		clear(t.data)
		return nil
	}
	return t.buf.Memset(0, t.buf.Size())
}

// Reshape returns a view with a new shape that shares the same buffer.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if Numel(shape) != t.Numel() {
		return nil, fmt.Errorf("tensor: reshape element count mismatch")
	}
	return &Tensor{buf: t.buf, data: t.data, device: t.device, shape: append([]int(nil), shape...)}, nil
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
