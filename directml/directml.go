// Package directml implements float32 inference using DirectML and D3D12.
// Drivers are loaded lazily. No SDK, cgo, shaders or extra Go modules are needed.
package directml

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/yh2237/gograd/gpu"
)

const defaultCacheLimit = 256 << 20

type Info struct {
	Name                             string
	VendorID, DeviceID, FeatureLevel uint32
	DedicatedVideoMemory             uint64
}

// AllocationStats covers tensor and staging resources, not driver-internal
// heaps or compiled operators. OperatorStats reports the separate bounded cache.
type AllocationStats struct {
	LiveBytes, CachedBytes, ReservedBytes uint64
	LiveBuffers, CachedBuffers            int
	CacheLimitBytes                       uint64
}

type OperatorStats struct {
	CachedOperators                 int
	PersistentBytes, TemporaryBytes uint64
}

type allocation struct {
	native    any
	size      uint64
	allocated uint64
	heap      uint32
}
type opKey struct {
	kind                      uint32
	m, n, k                   int
	ta, tb                    bool
	scalar                    uint32
	bias                      uint32
	batch                     int
	aLayout, bLayout, yLayout tensorLayout
	conv                      gpu.Conv1dOptions
	hasBias                   bool
}

// Four-dimensional descriptors cover the baseline DirectML feature level.
// Higher-rank public views are copied on the device before folding batch axes.
type tensorLayout struct{ sizes, strides [4]uint32 }

// resourceBinding offsets are bytes, aligned to DirectML's 16-byte tensor
// alignment. They are bindings, not cache keys: changing a slice reuses its op.
type resourceBinding struct {
	memory *allocation
	offset uint64
}

const (
	opAdd uint32 = iota
	opMul
	opScalar
	opGEMM
	opTanh
	opGELUExact
	opNorm
	opConv
	opSoftmax
	opIdentity
)
const (
	heapDefault  uint32 = 1
	heapUpload   uint32 = 2
	heapReadback uint32 = 3
)

// Driver methods are serialized by Backend.mu. Every submission waits for its
// fence before descriptors, command allocators or buffers are reused/released.
type driver interface {
	info() Info
	alloc(size uint64, heap uint32) (*allocation, error)
	free(*allocation)
	write(*allocation, []float32) error
	read(*allocation, []float32) error
	copy(out, src *allocation, size uint64) error
	execute(key opKey, out *allocation, inputs ...*allocation) error
	executeAt(key opKey, out *allocation, inputs ...resourceBinding) error
	copyView(out, src *allocation, v gpu.View, scatter bool) error
	operatorStats() OperatorStats
	releaseOperators()
	close()
}

type poolKey struct {
	size uint64
	heap uint32
}
type Backend struct {
	gpu.UnsupportedOps
	once               sync.Once
	mu                 sync.Mutex
	driver             driver
	err                error
	closed, registered bool
	pool               map[poolKey][]*allocation
	stats              AllocationStats
}

var _ gpu.Backend = (*Backend)(nil)
var _ gpu.Copier = (*Backend)(nil)
var _ gpu.Synchronizer = (*Backend)(nil)

func NewBackend() *Backend {
	return &Backend{stats: AllocationStats{CacheLimitBytes: defaultCacheLimit}}
}

var defaultBackend = NewBackend()

func init() {
	defaultBackend.registered = true
	if err := gpu.Register(defaultBackend); err != nil {
		panic(err)
	}
}
func (*Backend) Name() string { return "directml" }
func (b *Backend) lock() error {
	b.once.Do(func() {
		d, err := openDriver()
		if err != nil && !errors.Is(err, gpu.ErrUnavailable) {
			err = fmt.Errorf("%w: %v", gpu.ErrUnavailable, err)
		}
		b.mu.Lock()
		b.driver, b.err = d, err
		b.mu.Unlock()
	})
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return gpu.ErrClosed
	}
	if b.err != nil {
		err := b.err
		b.mu.Unlock()
		return err
	}
	return nil
}
func (b *Backend) Available() bool {
	if err := b.lock(); err != nil {
		return false
	}
	b.mu.Unlock()
	return true
}

// Synchronize waits behind active work. All current DirectML operations wait for
// their fence while holding mu; consequently no extra command list is needed.
func (b *Backend) Synchronize() error {
	if err := b.lock(); err != nil {
		return err
	}
	b.mu.Unlock()
	return nil
}
func (b *Backend) Err() error {
	if err := b.lock(); err != nil {
		return err
	}
	b.mu.Unlock()
	return nil
}
func (b *Backend) DeviceInfo() (Info, error) {
	if err := b.lock(); err != nil {
		return Info{}, err
	}
	defer b.mu.Unlock()
	return b.driver.info(), nil
}
func (b *Backend) MemoryStats() AllocationStats { b.mu.Lock(); defer b.mu.Unlock(); return b.stats }
func (b *Backend) OperatorStats() OperatorStats {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.driver == nil {
		return OperatorStats{}
	}
	return b.driver.operatorStats()
}
func Available() bool                    { return defaultBackend.Available() }
func Err() error                         { return defaultBackend.Err() }
func DeviceInfo() (Info, error)          { return defaultBackend.DeviceInfo() }
func MemoryStats() AllocationStats       { return defaultBackend.MemoryStats() }
func CachedOperatorStats() OperatorStats { return defaultBackend.OperatorStats() }
func ReleasePool()                       { defaultBackend.ReleasePool() }
func ReleaseOperators()                  { defaultBackend.ReleaseOperators() }
func SetPoolCacheLimit(bytes uint64)     { defaultBackend.SetPoolCacheLimit(bytes) }

func (b *Backend) releasePoolLocked() {
	for key, list := range b.pool {
		for _, v := range list {
			b.driver.free(v)
			b.stats.ReservedBytes -= v.allocated
		}
		delete(b.pool, key)
	}
	b.stats.CachedBytes, b.stats.CachedBuffers = 0, 0
}
func (b *Backend) ReleasePool() { b.mu.Lock(); defer b.mu.Unlock(); b.releasePoolLocked() }
func (b *Backend) ReleaseOperators() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.driver != nil {
		b.driver.releaseOperators()
	}
}
func (b *Backend) SetPoolCacheLimit(bytes uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.stats.CacheLimitBytes = bytes
	if b.stats.CachedBytes > bytes {
		b.releasePoolLocked()
	}
}
func (b *Backend) Close() error {
	if b.registered {
		return fmt.Errorf("%w: registered DirectML backend has process lifetime", gpu.ErrUnsupported)
	}
	b.once.Do(func() {})
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if b.stats.LiveBuffers != 0 {
		return fmt.Errorf("%w: DirectML backend still has live buffers", gpu.ErrInvalidArgument)
	}
	b.releasePoolLocked()
	if b.driver != nil {
		b.driver.close()
		b.driver = nil
	}
	b.closed = true
	return nil
}
func (b *Backend) fail(err error) error {
	if err != nil {
		b.err = fmt.Errorf("%w: DirectML device failure: %v", gpu.ErrUnavailable, err)
		b.releasePoolLocked()
		b.driver.releaseOperators()
	}
	return err
}
func sizeClass(size uint64) uint64 {
	n := uint64(256)
	for n < size {
		n <<= 1
	}
	return n
}
func (b *Backend) take(size uint64, heap uint32) (*allocation, error) {
	key := poolKey{sizeClass(size), heap}
	list := b.pool[key]
	var v *allocation
	if len(list) != 0 {
		v = list[len(list)-1]
		b.pool[key] = list[:len(list)-1]
		if len(list) == 1 {
			delete(b.pool, key)
		}
		b.stats.CachedBytes -= v.allocated
		b.stats.CachedBuffers--
	} else {
		var err error
		v, err = b.driver.alloc(key.size, heap)
		if err != nil {
			return nil, b.fail(err)
		}
		b.stats.ReservedBytes += v.allocated
	}
	b.stats.LiveBytes += v.allocated
	b.stats.LiveBuffers++
	return v, nil
}
func (b *Backend) put(v *allocation) {
	b.stats.LiveBytes -= v.allocated
	b.stats.LiveBuffers--
	if !b.closed && b.err == nil && v.allocated <= b.stats.CacheLimitBytes-b.stats.CachedBytes {
		if b.pool == nil {
			b.pool = make(map[poolKey][]*allocation)
		}
		key := poolKey{v.size, v.heap}
		b.pool[key] = append(b.pool[key], v)
		b.stats.CachedBytes += v.allocated
		b.stats.CachedBuffers++
	} else {
		b.driver.free(v)
		b.stats.ReservedBytes -= v.allocated
	}
}
func (b *Backend) native(buffer *gpu.Buffer, n int) (*allocation, error) {
	if err := gpu.CheckBuffer(b, buffer, n); err != nil {
		return nil, err
	}
	native, err := buffer.Native(b)
	if err != nil {
		return nil, err
	}
	v, ok := native.(*allocation)
	if !ok || v == nil || v.heap != heapDefault || v.size < uint64(n)*4 {
		return nil, gpu.ErrInvalidArgument
	}
	return v, nil
}
func (b *Backend) Alloc(n int) (*gpu.Buffer, error) {
	if _, err := gpu.Elements(n); err != nil {
		return nil, err
	}
	if err := b.lock(); err != nil {
		return nil, err
	}
	defer b.mu.Unlock()
	v, err := b.take(uint64(n)*4, heapDefault)
	if err != nil {
		return nil, err
	}
	out, err := gpu.NewBuffer(b, n, v, func() { b.mu.Lock(); defer b.mu.Unlock(); b.put(v) })
	if err != nil {
		b.put(v)
	}
	return out, err
}
func (b *Backend) Release(buffer *gpu.Buffer) {
	if buffer != nil && buffer.Backend() != b {
		panic(gpu.ErrBackendMismatch)
	}
	buffer.Close()
}
func (b *Backend) Upload(values []float32) (*gpu.Buffer, error) {
	out, err := b.Alloc(len(values))
	if err != nil {
		return nil, err
	}
	if err := b.CopyFrom(out, values); err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}
func (b *Backend) CopyFrom(buffer *gpu.Buffer, values []float32) error {
	return b.upload(buffer, values, false)
}
func (b *Backend) Zero(buffer *gpu.Buffer) error { return b.upload(buffer, nil, true) }
func (b *Backend) upload(buffer *gpu.Buffer, values []float32, zero bool) error {
	if buffer == nil {
		return gpu.ErrClosed
	}
	n := len(values)
	if zero {
		n = buffer.Len()
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	out, err := b.native(buffer, n)
	if err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	staging, err := b.take(uint64(n)*4, heapUpload)
	if err != nil {
		return err
	}
	defer b.put(staging)
	if err := b.fail(b.driver.write(staging, values)); err != nil {
		return err
	}
	return b.fail(b.driver.copy(out, staging, uint64(n)*4))
}
func (b *Backend) Download(buffer *gpu.Buffer, n int) ([]float32, error) {
	if buffer == nil {
		return nil, gpu.ErrClosed
	}
	if n < 0 || n > buffer.Len() {
		return nil, gpu.ErrInvalidArgument
	}
	if err := b.lock(); err != nil {
		return nil, err
	}
	defer b.mu.Unlock()
	x, err := b.native(buffer, buffer.Len())
	if err != nil {
		return nil, err
	}
	values := make([]float32, n)
	if n == 0 {
		return values, nil
	}
	staging, err := b.take(uint64(n)*4, heapReadback)
	if err != nil {
		return nil, err
	}
	defer b.put(staging)
	if err := b.fail(b.driver.copy(staging, x, uint64(n)*4)); err != nil {
		return nil, err
	}
	if err := b.fail(b.driver.read(staging, values)); err != nil {
		return nil, err
	}
	return values, nil
}
func (b *Backend) Copy(out, src *gpu.Buffer, n int) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(src, n)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	if x == y {
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	return b.fail(b.driver.copy(y, x, uint64(n)*4))
}
func (b *Backend) pointwise(out, a, right *gpu.Buffer, n int, kind uint32, scalar float32) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(a, n)
	if err != nil {
		return err
	}
	w, err := b.native(right, n)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	if y == x || y == w {
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	return b.fail(b.driver.execute(opKey{kind: kind, n: n, scalar: math.Float32bits(scalar)}, y, x, w))
}
func (b *Backend) Add(out, a, right *gpu.Buffer, n int) error {
	return b.pointwise(out, a, right, n, opAdd, 0)
}
func (b *Backend) Mul(out, a, right *gpu.Buffer, n int) error {
	return b.pointwise(out, a, right, n, opMul, 0)
}
func (b *Backend) MulScalar(out, a *gpu.Buffer, n int, scalar float32) error {
	return b.pointwise(out, a, a, n, opScalar, scalar)
}
func (b *Backend) MatMul(out, a, right *gpu.Buffer, m, n, k int, ta, tb bool) error {
	mn, err := gpu.Elements(m, n)
	if err != nil {
		return err
	}
	mk, err := gpu.Elements(m, k)
	if err != nil {
		return err
	}
	kn, err := gpu.Elements(k, n)
	if err != nil {
		return err
	}
	if err := b.lock(); err != nil {
		return err
	}
	x, err := b.native(a, mk)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	w, err := b.native(right, kn)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	y, err := b.native(out, mn)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	if y == x || y == w {
		b.mu.Unlock()
		return gpu.ErrInvalidArgument
	}
	if mn == 0 {
		b.mu.Unlock()
		return nil
	}
	if k == 0 {
		b.mu.Unlock()
		return b.Zero(out)
	}
	defer b.mu.Unlock()
	return b.fail(b.driver.execute(opKey{kind: opGEMM, m: m, n: n, k: k, ta: ta, tb: tb}, y, x, w))
}
