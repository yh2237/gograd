// Package gpu defines the float32 inference backend contract. It does not load
// drivers, select a CPU fallback, or depend on autograd, tensor, cgo or a GPU SDK.
package gpu

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"
)

var (
	ErrUnavailable     = errors.New("gpu: backend unavailable")
	ErrUnsupported     = errors.New("gpu: operation unsupported")
	ErrInvalidArgument = errors.New("gpu: invalid argument")
	ErrClosed          = errors.New("gpu: buffer is closed")
	ErrBackendMismatch = errors.New("gpu: buffer belongs to another backend")
	ErrInferenceOnly   = errors.New("gpu: backend supports inference only")
)

// Backend executes contiguous, row-major float32 forward operations. Sizes are
// element counts, not bytes. Operations enqueue work in backend order; Download
// must wait for the values it reads. Release must defer reuse until queued work
// using that allocation is safe. Backends must return ErrUnsupported rather than
// silently downloading operands and evaluating an operation on the CPU.
// Outputs must be distinct from their inputs. Shapes/counts must match the
// complete logical buffers, except that Download may read a prefix.
//
// Implementations must have comparable identity (normally a pointer). A buffer
// belongs to the specific instance that allocated it, not just its backend name.
// Name must remain constant. A registered implementation must remain usable for
// the process lifetime.
type Backend interface {
	Name() string
	Available() bool
	Alloc(n int) (*Buffer, error)
	Upload(values []float32) (*Buffer, error)
	Download(b *Buffer, n int) ([]float32, error)
	CopyFrom(b *Buffer, values []float32) error
	Zero(b *Buffer) error
	Release(b *Buffer)

	MatMul(out, a, b *Buffer, m, n, k int, transA, transB bool) error
	Add(out, a, b *Buffer, n int) error
	Mul(out, a, b *Buffer, n int) error
	MulScalar(out, a *Buffer, n int, scalar float32) error
	// GELU uses the tanh approximation on all inference backends.
	GELU(out, a *Buffer, n int) error
	Softmax(out, a *Buffer, rows, cols int) error
	LayerNorm(out, a, weight, bias *Buffer, rows, cols int, eps float32) error
	// GroupNorm uses [batch,frames,channels], with affine vectors [channels].
	GroupNorm(out, a, weight, bias *Buffer, batch, frames, channels, groups int, eps float32) error
	Conv1d(out, x, weight, bias *Buffer, options Conv1dOptions) error
}

// Conv1dOptions describes cross-correlation with [batch,time,in] input and
// [out,in,kernel] weights. Bias may be nil. Padding is symmetric; zero stride or
// dilation is invalid (there are no implicit defaults in the backend ABI).
type Conv1dOptions struct {
	Batch, InputLength, InChannels, OutChannels, Kernel int
	Stride, Padding, Dilation                           int
}

func (o Conv1dOptions) OutputLength() (int, error) {
	for _, d := range []int{o.Batch, o.InputLength, o.InChannels, o.OutChannels, o.Kernel} {
		if _, err := Elements(d); err != nil {
			return 0, err
		}
	}
	if o.Batch < 1 || o.InputLength < 1 || o.InChannels < 1 || o.OutChannels < 1 || o.Kernel < 1 || o.Stride < 1 || o.Dilation < 1 || o.Padding < 0 || o.Stride > maxElements || o.Dilation > maxElements || o.Padding > maxElements {
		return 0, fmt.Errorf("%w: Conv1d dimensions", ErrInvalidArgument)
	}
	span := int64(o.Dilation)*int64(o.Kernel-1) + 1
	padded := int64(o.InputLength) + 2*int64(o.Padding)
	if span > int64(maxElements) || padded > int64(maxElements) {
		return 0, fmt.Errorf("%w: Conv1d index range is too large", ErrInvalidArgument)
	}
	if span > padded {
		return 0, fmt.Errorf("%w: Conv1d effective kernel exceeds input", ErrInvalidArgument)
	}
	length := (padded-span)/int64(o.Stride) + 1
	if length > int64(maxElements) {
		return 0, fmt.Errorf("%w: Conv1d output is too large", ErrInvalidArgument)
	}
	return int(length), nil
}

const maxElements = min(1<<31-1, int(^uint(0)>>1)/4)

// Elements validates GPU dimensions and their product before allocations or
// conversion to kernel int32 indices. A scalar has one element; zero axes are
// allowed. The byte size of the resulting float32 allocation must fit in int.
func Elements(shape ...int) (int, error) {
	zero := false
	for _, d := range shape {
		if d < 0 || d > maxElements {
			return 0, fmt.Errorf("%w: GPU dimensions %v", ErrInvalidArgument, shape)
		}
		zero = zero || d == 0
	}
	if zero {
		return 0, nil
	}
	n := 1
	for _, d := range shape {
		if n > maxElements/d {
			return 0, fmt.Errorf("%w: GPU dimensions overflow %v", ErrInvalidArgument, shape)
		}
		n *= d
	}
	return n, nil
}

func validateBackend(b Backend) error {
	if b == nil {
		return fmt.Errorf("%w: nil backend", ErrInvalidArgument)
	}
	v := reflect.ValueOf(b)
	if !v.Comparable() || (v.Kind() == reflect.Pointer && v.IsNil()) {
		return fmt.Errorf("%w: backend needs non-nil comparable identity", ErrInvalidArgument)
	}
	return nil
}

// Registry is safe for concurrent lookup/registration. The zero value is ready
// for use. Registrations cannot be replaced: live buffers keep their owner.
type Registry struct {
	mu       sync.RWMutex
	backends map[string]Backend
}

func (r *Registry) Register(b Backend) error {
	if err := validateBackend(b); err != nil {
		return err
	}
	name := b.Name()
	if name == "" || name == "cpu" {
		return fmt.Errorf("%w: reserved or empty backend name %q", ErrInvalidArgument, name)
	}
	for _, c := range name {
		if c < 'a' || c > 'z' {
			return fmt.Errorf("%w: backend name must contain lowercase ASCII letters", ErrInvalidArgument)
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.backends[name]; exists {
		return fmt.Errorf("%w: backend %q is already registered", ErrInvalidArgument, name)
	}
	if r.backends == nil {
		r.backends = make(map[string]Backend)
	}
	r.backends[name] = b
	return nil
}

func (r *Registry) Lookup(name string) (Backend, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	b, ok := r.backends[name]
	return b, ok
}

// Get requires an available backend. It never chooses a different device.
func (r *Registry) Get(name string) (Backend, error) {
	b, ok := r.Lookup(name)
	if !ok || !b.Available() {
		return nil, fmt.Errorf("%w: %s", ErrUnavailable, name)
	}
	return b, nil
}

func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.backends))
	for name := range r.backends {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

var defaultRegistry Registry

func Register(b Backend) error           { return defaultRegistry.Register(b) }
func Lookup(name string) (Backend, bool) { return defaultRegistry.Lookup(name) }
func Get(name string) (Backend, error)   { return defaultRegistry.Get(name) }
func Names() []string                    { return defaultRegistry.Names() }
func Available(name string) bool         { _, err := Get(name); return err == nil }
