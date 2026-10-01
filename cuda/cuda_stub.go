//go:build !windows

// Package cuda binds the CUDA runtime and cuBLAS DLLs on Windows. On other
// platforms the API exists but every call reports ErrUnavailable.
package cuda

import (
	"errors"
	"unsafe"
)

// ErrUnavailable reports that this platform has no CUDA binding.
var ErrUnavailable = errors.New("cuda: runtime binding is Windows-only")

// Error is a CUDA or cuBLAS failure.
type Error struct {
	Op      string
	Code    int
	Message string
}

func (e *Error) Error() string { return "cuda: " + e.Op + ": " + e.Message }

// Available always reports false on this platform.
func Available() bool { return false }

// DeviceCount returns ErrUnavailable.
func DeviceCount() (int, error) { return 0, ErrUnavailable }

// SetDevice returns ErrUnavailable.
func SetDevice(device int) error { return ErrUnavailable }

// CurrentDevice returns ErrUnavailable.
func CurrentDevice() (int, error) { return 0, ErrUnavailable }

// Synchronize returns ErrUnavailable.
func Synchronize() error { return ErrUnavailable }

// LastError returns ErrUnavailable.
func LastError() error { return ErrUnavailable }

// Buffer is a device allocation.
type Buffer struct {
	pointer uintptr
	size    int
}

// Alloc returns ErrUnavailable.
func Alloc(size int) (*Buffer, error) { return nil, ErrUnavailable }

// Free returns ErrUnavailable.
func (b *Buffer) Free() error { return ErrUnavailable }

// ReleasePool returns nil on this platform.
func ReleasePool() error { return nil }

// SetCurrentStream does nothing on this platform.
func SetCurrentStream(s *Stream) {}

// InCapture always reports false on this platform.
func InCapture() bool { return false }

// Graph is a captured sequence of GPU work.
type Graph struct{ exec uintptr }

// Capture returns ErrUnavailable.
func Capture(stream *Stream, fn func() error) (*Graph, error) { return nil, ErrUnavailable }

// Launch returns ErrUnavailable.
func (g *Graph) Launch() error { return ErrUnavailable }

// Close returns ErrUnavailable.
func (g *Graph) Close() error { return ErrUnavailable }

// Pointer returns the raw device address.
func (b *Buffer) Pointer() uintptr { return 0 }

// Size returns the allocation size in bytes.
func (b *Buffer) Size() int { return 0 }

// CopyFromHost returns ErrUnavailable.
func (b *Buffer) CopyFromHost(data []byte) error { return ErrUnavailable }

// CopyToHost returns ErrUnavailable.
func (b *Buffer) CopyToHost(data []byte) error { return ErrUnavailable }

// Memset returns ErrUnavailable.
func (b *Buffer) Memset(value byte, size int) error { return ErrUnavailable }

// MemsetAsync returns ErrUnavailable.
func (b *Buffer) MemsetAsync(value byte, size int) error { return ErrUnavailable }

// Stream is a CUDA stream.
type Stream struct{ handle uintptr }

// NewStream returns ErrUnavailable.
func NewStream() (*Stream, error) { return nil, ErrUnavailable }

// Synchronize returns ErrUnavailable.
func (s *Stream) Synchronize() error { return ErrUnavailable }

// Destroy returns ErrUnavailable.
func (s *Stream) Destroy() error { return ErrUnavailable }

// Event is a CUDA event.
type Event struct{ handle uintptr }

// NewEvent returns ErrUnavailable.
func NewEvent(timing bool) (*Event, error) { return nil, ErrUnavailable }

// Record returns ErrUnavailable.
func (e *Event) Record(stream *Stream) error { return ErrUnavailable }

// Synchronize returns ErrUnavailable.
func (e *Event) Synchronize() error { return ErrUnavailable }

// Destroy returns ErrUnavailable.
func (e *Event) Destroy() error { return ErrUnavailable }

// Blas is a cuBLAS handle.
type Blas struct{ handle uintptr }

// NewBlas returns ErrUnavailable.
func NewBlas() (*Blas, error) { return nil, ErrUnavailable }

// SetStream returns ErrUnavailable.
func (b *Blas) SetStream(stream *Stream) error { return ErrUnavailable }

// Destroy returns ErrUnavailable.
func (b *Blas) Destroy() error { return ErrUnavailable }

// SgemmRowMajor returns ErrUnavailable.
func (b *Blas) SgemmRowMajor(m, n, k int, alpha float32, a uintptr, lda int, bPtr uintptr, ldb int, beta float32, c uintptr, ldc int) error {
	return ErrUnavailable
}

// SgemmRowMajorNT returns ErrUnavailable.
func (b *Blas) SgemmRowMajorNT(m, n, k int, alpha float32, a uintptr, lda int, bPtr uintptr, ldb int, beta float32, c uintptr, ldc int) error {
	return ErrUnavailable
}

// SgemmRowMajorTransposeA returns ErrUnavailable.
func (b *Blas) SgemmRowMajorTransposeA(m, n, k int, alpha float32, a uintptr, lda int, bPtr uintptr, ldb int, beta float32, c uintptr, ldc int) error {
	return ErrUnavailable
}

// SgemmStridedBatchedRowMajor returns ErrUnavailable.
func (b *Blas) SgemmStridedBatchedRowMajor(batchCount, m, n, k int, alpha float32, a uintptr, lda int, strideA int64, bPtr uintptr, ldb int, strideB int64, beta float32, c uintptr, ldc int, strideC int64) error {
	return ErrUnavailable
}

// DeviceAttribute returns ErrUnavailable.
func DeviceAttribute(attribute, device int) (int, error) { return 0, ErrUnavailable }

// Kernel is a compiled device function.
type Kernel struct{ module, function uintptr }

// Program is a compiled module.
type Program struct{ module uintptr }

// Compile returns ErrUnavailable.
func Compile(source string) (*Program, error) { return nil, ErrUnavailable }

// Function returns ErrUnavailable.
func (p *Program) Function(name string) (*Kernel, error) { return nil, ErrUnavailable }

// Close returns ErrUnavailable.
func (p *Program) Close() error { return ErrUnavailable }

// LoadKernel returns ErrUnavailable.
func LoadKernel(source, name string) (*Kernel, error) { return nil, ErrUnavailable }

// Launch returns ErrUnavailable.
func (k *Kernel) Launch(grid, block [3]int, sharedMemory int, stream *Stream, args []unsafe.Pointer) error {
	return ErrUnavailable
}

// Close returns ErrUnavailable.
func (k *Kernel) Close() error { return ErrUnavailable }
