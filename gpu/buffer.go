package gpu

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// Buffer owns one backend allocation. Do not copy a Buffer or release its native
// handle independently. Close/Backend.Release are idempotent. Like CUDA buffers,
// concurrent access and Close require synchronization by the caller.
type Buffer struct {
	owner  Backend
	native any
	n      int
	free   func()
	once   sync.Once
	closed atomic.Bool
}

// NewBuffer transfers an allocation to a backend-neutral owner. On error the
// caller still owns native. free must release exactly that allocation, safely
// with respect to already-enqueued work on the backend.
func NewBuffer(owner Backend, n int, native any, free func()) (*Buffer, error) {
	if err := validateBackend(owner); err != nil {
		return nil, err
	}
	if _, err := Elements(n); err != nil {
		return nil, err
	}
	if free == nil {
		return nil, fmt.Errorf("%w: missing buffer release function", ErrInvalidArgument)
	}
	return &Buffer{owner: owner, native: native, n: n, free: free}, nil
}

func (b *Buffer) Len() int         { return b.n }
func (b *Buffer) Backend() Backend { return b.owner }

// Native borrows the backend-specific allocation after checking its owner. It
// must not be kept after Close or passed to a different backend instance.
func (b *Buffer) Native(owner Backend) (any, error) {
	if b == nil || b.closed.Load() {
		return nil, ErrClosed
	}
	if err := validateBackend(owner); err != nil {
		return nil, err
	}
	if b.owner != owner {
		return nil, ErrBackendMismatch
	}
	return b.native, nil
}

func (b *Buffer) Close() {
	if b == nil {
		return
	}
	b.once.Do(func() {
		b.closed.Store(true)
		b.free()
	})
}

// CheckBuffer validates ownership, lifetime and the exact logical element
// count. Backends use it before passing untrusted dimensions to a driver.
func CheckBuffer(owner Backend, b *Buffer, n int) error {
	if _, err := Elements(n); err != nil {
		return err
	}
	if _, err := b.Native(owner); err != nil {
		return err
	}
	if b.n != n {
		return fmt.Errorf("%w: buffer has %d elements, expected %d", ErrInvalidArgument, b.n, n)
	}
	return nil
}
