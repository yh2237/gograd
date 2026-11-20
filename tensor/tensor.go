// Package tensor provides contiguous float32 tensors on CPU or CUDA.
package tensor

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/gpu"
)

// Tensor is a contiguous, row-major float32 device tensor. Use it by pointer.
type Tensor struct {
	storage *cudaStorage
	cleanup runtime.Cleanup
	data    []float32
	device  Device
	shape   []int
	closed  bool
}

type cudaStorage struct {
	buffer *cuda.Buffer
	refs   atomic.Int64
}

func (s *cudaStorage) release() {
	if s.refs.Add(-1) == 0 {
		s.buffer.Free()
	}
}

// Device selects storage at construction time.
type Device string

const (
	CPU  Device = "cpu"
	CUDA Device = "cuda"
	// Vulkan and DirectML are reserved for the inference backend rollout.
	// Their presence is not an availability guarantee; consult gpu.Available.
	Vulkan   Device = "vulkan"
	DirectML Device = "directml"
)

// Numel returns the number of elements described by shape.
func Numel(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

func checkedNumel(shape []int) (int, error) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("tensor: negative dimension")
		}
		if d != 0 && n > math.MaxInt/d {
			return 0, fmt.Errorf("tensor: oversized shape %v", shape)
		}
		n *= d
	}
	return n, nil
}

// New allocates an uninitialized tensor with the given shape.
func New(shape ...int) (*Tensor, error) {
	return NewOn(CUDA, shape...)
}

// NewOn allocates a tensor on the chosen device.
func NewOn(device Device, shape ...int) (*Tensor, error) {
	if device == Vulkan || device == DirectML {
		return nil, fmt.Errorf("%w: %s tensor integration is not implemented yet", gpu.ErrUnavailable, device)
	}
	n, err := checkedNumel(shape)
	if err != nil {
		return nil, err
	}
	if n > math.MaxInt/4 {
		return nil, fmt.Errorf("tensor: oversized shape %v", shape)
	}
	if device == CPU {
		return &Tensor{data: make([]float32, n), device: CPU, shape: append([]int(nil), shape...)}, nil
	}
	if device != CUDA {
		return nil, fmt.Errorf("tensor: unknown device %q", device)
	}
	buf, err := cuda.Alloc(n * 4)
	if err != nil {
		return nil, err
	}
	storage := &cudaStorage{buffer: buf}
	storage.refs.Store(1)
	return &Tensor{storage: storage, device: CUDA, shape: append([]int(nil), shape...)}, nil
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

// Buffer borrows the device buffer. Keep the tensor reachable while using it
// and release it through Tensor.Close. CPU tensors return nil.
func (t *Tensor) Buffer() *cuda.Buffer {
	if t.storage == nil {
		return nil
	}
	return t.storage.buffer
}

// Device returns the storage device.
func (t *Tensor) Device() Device { return t.device }

// Data returns the CPU backing slice, or nil for CUDA or closed tensors.
func (t *Tensor) Data() []float32 {
	if t.closed {
		return nil
	}
	return t.data
}

// ToHost downloads the tensor.
func (t *Tensor) ToHost() ([]float32, error) {
	if t.closed {
		return nil, fmt.Errorf("tensor: tensor is closed")
	}
	if t.device == CPU {
		return append([]float32(nil), t.data...), nil
	}
	out := make([]float32, t.Numel())
	err := t.storage.buffer.CopyToHost(bytesOf(out))
	runtime.KeepAlive(t)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// CopyFrom uploads host data into the tensor.
func (t *Tensor) CopyFrom(data []float32) error {
	if t.closed {
		return fmt.Errorf("tensor: tensor is closed")
	}
	if len(data) != t.Numel() {
		return fmt.Errorf("tensor: data length %d does not match %d elements", len(data), t.Numel())
	}
	if t.device == CPU {
		copy(t.data, data)
		return nil
	}
	err := t.storage.buffer.CopyFromHost(bytesOf(data))
	runtime.KeepAlive(t)
	return err
}

// Zero clears the tensor.
func (t *Tensor) Zero() error {
	if t.closed {
		return fmt.Errorf("tensor: tensor is closed")
	}
	if t.device == CPU {
		clear(t.data)
		return nil
	}
	buf := t.storage.buffer
	err := buf.Memset(0, buf.Size())
	runtime.KeepAlive(t)
	return err
}

// Reshape shares the allocation with a new shape. Close the returned handle
// to release its share of the allocation.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if t.closed {
		return nil, fmt.Errorf("tensor: tensor is closed")
	}
	n, err := checkedNumel(shape)
	if err != nil {
		return nil, err
	}
	if n != t.Numel() {
		return nil, fmt.Errorf("tensor: reshape element count mismatch")
	}
	view := &Tensor{storage: t.storage, data: t.data, device: t.device, shape: append([]int(nil), shape...)}
	if view.storage != nil {
		view.storage.refs.Add(1)
		// Older callers may let a borrowed view go out of scope without Close.
		view.cleanup = runtime.AddCleanup(view, (*cudaStorage).release, view.storage)
		runtime.KeepAlive(t)
	}
	return view, nil
}

// Close ends this handle. CUDA memory returns to the pool when the final
// handle releases the allocation.
func (t *Tensor) Close() {
	if t.closed {
		return
	}
	t.closed = true
	t.data = nil
	if storage := t.storage; storage != nil {
		t.cleanup.Stop()
		t.storage = nil
		storage.release()
		runtime.KeepAlive(t)
	}
}

func bytesOf(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
