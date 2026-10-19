// Package vulkan provides a headless, float32 inference backend. Driver loading
// is lazy; shader binaries are embedded, so users do not need a Vulkan SDK.
package vulkan

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

const defaultCacheLimit = 256 << 20

type Info struct {
	Name                                        string
	APIVersion, VendorID, DeviceID, QueueFamily uint32
	MaxStorageBufferBytes                       uint64
	MaxWorkGroupCount                           [3]uint32
}

// AllocationStats counts backing allocations, including temporary staging
// buffers. Cached memory is bounded and can be returned with ReleasePool.
// Quarantined memory remains reserved until Close proves completion or device loss.
type AllocationStats struct {
	LiveBytes, CachedBytes, ReservedBytes uint64
	LiveBuffers, CachedBuffers            int
	QuarantinedBytes                      uint64
	QuarantinedBuffers                    int
	CacheLimitBytes                       uint64
}

type block struct {
	buffer, memory, size, allocated uint64
	mapped                          unsafe.Pointer
	host                            bool
}

// All driver calls are serialized by Backend.mu, including queue, descriptors,
// command pool and fence use. Every public operation waits for completion.
type driver interface {
	info() Info
	alloc(size uint64, host bool) (*block, error)
	free(*block) bool // false transfers ownership to the driver's quarantine
	write(*block, []float32)
	read(*block, []float32)
	copy(out, src *block, size uint64) error
	zero(out *block, size uint64) error
	pointwise(out, a, b *block, n, kind uint32, scalar float32) error
	fusedBias(out, a, bias, residual *block, n, channels, kind uint32) error
	viewCopy(out, src *block, outCount, srcCount uint32, params viewParams) error
	binaryViews(out, a, b *block, outCount, aCount, bCount uint32, params binaryParams) error
	gemm(out, a, b *block, outCount, aCount, bCount uint32, params gemmParams, variant int) error
	reduce(out, a, weight, bias *block, count, affineCount uint32, params reduceParams) error
	conv1d(convBatch) error
	close() error
}

type poolKey struct {
	size uint64
	host bool
}

// Backend owns one logical device and queue. NewBackend creates an independent
// instance; the registered default backend lives for the process lifetime.
type Backend struct {
	gpu.UnsupportedOps
	once               sync.Once
	mu                 sync.Mutex
	driver             driver
	err                error
	closed, registered bool
	pool               map[poolKey][]*block
	stats              AllocationStats
}

var _ gpu.Backend = (*Backend)(nil)
var _ gpu.Copier = (*Backend)(nil)
var _ gpu.ViewBackend = (*Backend)(nil)
var _ gpu.ViewMatMuler = (*Backend)(nil)
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

func (*Backend) Name() string { return "vulkan" }

// lock returns with mu held only on success. Initialization errors are retained
// and exposed by Err instead of pretending a partially initialized device works.
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

// Synchronize waits behind any active operation. Every Vulkan operation already
// waits for its fence before releasing mu, so no extra submission is necessary.
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
func Available() bool                { return defaultBackend.Available() }
func Err() error                     { return defaultBackend.Err() }
func DeviceInfo() (Info, error)      { return defaultBackend.DeviceInfo() }
func MemoryStats() AllocationStats   { return defaultBackend.MemoryStats() }
func ReleasePool()                   { defaultBackend.ReleasePool() }
func SetPoolCacheLimit(bytes uint64) { defaultBackend.SetPoolCacheLimit(bytes) }

func (b *Backend) MemoryStats() AllocationStats { b.mu.Lock(); defer b.mu.Unlock(); return b.stats }
func (b *Backend) releasePoolLocked() {
	for key, blocks := range b.pool {
		for _, block := range blocks {
			b.freeLocked(block)
		}
		delete(b.pool, key)
	}
	b.stats.CachedBytes, b.stats.CachedBuffers = 0, 0
}
func (b *Backend) ReleasePool() { b.mu.Lock(); defer b.mu.Unlock(); b.releasePoolLocked() }
func (b *Backend) SetPoolCacheLimit(bytes uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.stats.CacheLimitBytes = bytes
	if b.stats.CachedBytes > bytes {
		b.releasePoolLocked()
	}
}

// Close is for independent backends only. Live buffers must be closed first.
// If failed work cannot be proven complete, Close retains quarantined resources
// and returns an error. A later Close makes one more idle attempt; it never spins.
func (b *Backend) Close() error {
	if b.registered {
		return fmt.Errorf("%w: registered Vulkan backend has process lifetime", gpu.ErrUnsupported)
	}
	b.once.Do(func() {})
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if b.stats.LiveBuffers != 0 {
		return fmt.Errorf("%w: Vulkan backend still has live buffers", gpu.ErrInvalidArgument)
	}
	b.releasePoolLocked()
	if b.driver != nil {
		if err := b.driver.close(); err != nil {
			return err // Retain driver ownership; a subsequent Close may retry once.
		}
	}
	b.stats.ReservedBytes, b.stats.QuarantinedBytes, b.stats.QuarantinedBuffers = 0, 0, 0
	b.closed = true
	return nil
}

func (b *Backend) fail(err error) error {
	if err != nil {
		b.err = fmt.Errorf("%w: Vulkan device failure: %w", gpu.ErrUnavailable, err)
		b.releasePoolLocked()
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

func dispatchGrid(n uint32, limits [3]uint32) (uint32, uint32, error) {
	groups := (uint64(n) + 127) / 128
	return groupGrid(groups, limits)
}

func groupGrid(groups uint64, limits [3]uint32) (uint32, uint32, error) {
	x := min(groups, uint64(limits[0]))
	if x == 0 {
		return 0, 0, fmt.Errorf("%w: Vulkan dispatch group limit", gpu.ErrUnsupported)
	}
	y := (groups + x - 1) / x
	if y > uint64(limits[1]) {
		return 0, 0, fmt.Errorf("%w: Vulkan dispatch exceeds group limits", gpu.ErrUnsupported)
	}
	return uint32(x), uint32(y), nil
}
func (b *Backend) take(size uint64, host bool) (*block, error) {
	key := poolKey{sizeClass(size), host}
	list := b.pool[key]
	var memory *block
	if len(list) != 0 {
		memory = list[len(list)-1]
		b.pool[key] = list[:len(list)-1]
		if len(list) == 1 {
			delete(b.pool, key)
		}
		b.stats.CachedBytes -= memory.allocated
		b.stats.CachedBuffers--
	} else {
		var err error
		memory, err = b.driver.alloc(key.size, host)
		if err != nil {
			return nil, b.fail(err)
		}
		b.stats.ReservedBytes += memory.allocated
	}
	b.stats.LiveBytes += memory.allocated
	b.stats.LiveBuffers++
	return memory, nil
}
func (b *Backend) put(memory *block) {
	b.stats.LiveBytes -= memory.allocated
	b.stats.LiveBuffers--
	if !b.closed && b.err == nil && memory.allocated <= b.stats.CacheLimitBytes-b.stats.CachedBytes {
		if b.pool == nil {
			b.pool = make(map[poolKey][]*block)
		}
		key := poolKey{memory.size, memory.host}
		b.pool[key] = append(b.pool[key], memory)
		b.stats.CachedBytes += memory.allocated
		b.stats.CachedBuffers++
	} else {
		b.freeLocked(memory)
	}
}

func (b *Backend) freeLocked(memory *block) {
	if b.driver.free(memory) {
		b.stats.ReservedBytes -= memory.allocated
	} else {
		b.stats.QuarantinedBytes += memory.allocated
		b.stats.QuarantinedBuffers++
	}
}
func (b *Backend) native(buffer *gpu.Buffer, n int) (*block, error) {
	if err := gpu.CheckBuffer(b, buffer, n); err != nil {
		return nil, err
	}
	handle, err := buffer.Native(b)
	if err != nil {
		return nil, err
	}
	memory, ok := handle.(*block)
	if !ok || memory == nil || memory.host || memory.size < uint64(n)*4 {
		return nil, fmt.Errorf("%w: invalid Vulkan allocation", gpu.ErrInvalidArgument)
	}
	return memory, nil
}

func (b *Backend) Alloc(n int) (*gpu.Buffer, error) {
	if _, err := gpu.Elements(n); err != nil {
		return nil, err
	}
	if err := b.lock(); err != nil {
		return nil, err
	}
	defer b.mu.Unlock()
	if uint64(n)*4 > b.driver.info().MaxStorageBufferBytes {
		return nil, fmt.Errorf("%w: exceeds Vulkan storage buffer range", gpu.ErrUnsupported)
	}
	memory, err := b.take(uint64(n)*4, false)
	if err != nil {
		return nil, err
	}
	buffer, err := gpu.NewBuffer(b, n, memory, func() { b.mu.Lock(); defer b.mu.Unlock(); b.put(memory) })
	if err != nil {
		b.put(memory)
	}
	return buffer, err
}
func (b *Backend) Release(buffer *gpu.Buffer) {
	if buffer != nil && buffer.Backend() != b {
		panic(gpu.ErrBackendMismatch)
	}
	buffer.Close()
}
func (b *Backend) Upload(values []float32) (*gpu.Buffer, error) {
	buffer, err := b.Alloc(len(values))
	if err != nil {
		return nil, err
	}
	if err := b.CopyFrom(buffer, values); err != nil {
		buffer.Close()
		return nil, err
	}
	return buffer, nil
}
func (b *Backend) CopyFrom(buffer *gpu.Buffer, values []float32) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	memory, err := b.native(buffer, len(values))
	if err != nil {
		return err
	}
	if len(values) == 0 {
		return nil
	}
	staging, err := b.take(uint64(len(values))*4, true)
	if err != nil {
		return err
	}
	defer b.put(staging)
	b.driver.write(staging, values)
	return b.fail(b.driver.copy(memory, staging, uint64(len(values))*4))
}
func (b *Backend) Download(buffer *gpu.Buffer, n int) ([]float32, error) {
	if buffer == nil {
		return nil, gpu.ErrClosed
	}
	if n < 0 || n > buffer.Len() {
		return nil, fmt.Errorf("%w: download length", gpu.ErrInvalidArgument)
	}
	if err := b.lock(); err != nil {
		return nil, err
	}
	defer b.mu.Unlock()
	memory, err := b.native(buffer, buffer.Len())
	if err != nil {
		return nil, err
	}
	values := make([]float32, n)
	if n == 0 {
		return values, nil
	}
	staging, err := b.take(uint64(n)*4, true)
	if err != nil {
		return nil, err
	}
	defer b.put(staging)
	if err := b.fail(b.driver.copy(staging, memory, uint64(n)*4)); err != nil {
		return nil, err
	}
	b.driver.read(staging, values)
	return values, nil
}
func (b *Backend) Zero(buffer *gpu.Buffer) error {
	if buffer == nil {
		return gpu.ErrClosed
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	memory, err := b.native(buffer, buffer.Len())
	if err != nil {
		return err
	}
	if buffer.Len() == 0 {
		return nil
	}
	return b.fail(b.driver.zero(memory, uint64(buffer.Len())*4))
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
		return fmt.Errorf("%w: output aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	return b.fail(b.driver.copy(y, x, uint64(n)*4))
}
func (b *Backend) elementwise(out, a, right *gpu.Buffer, n int, kind uint32, scalar float32) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(a, n)
	if err != nil {
		return err
	}
	y, err := b.native(right, n)
	if err != nil {
		return err
	}
	z, err := b.native(out, n)
	if err != nil {
		return err
	}
	if z == x || z == y {
		return fmt.Errorf("%w: output aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	if _, _, err := dispatchGrid(uint32(n), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	return b.fail(b.driver.pointwise(z, x, y, uint32(n), kind, scalar))
}
func (b *Backend) Add(out, a, right *gpu.Buffer, n int) error {
	return b.elementwise(out, a, right, n, 0, 0)
}
func (b *Backend) Mul(out, a, right *gpu.Buffer, n int) error {
	return b.elementwise(out, a, right, n, 1, 0)
}
func (b *Backend) MulScalar(out, a *gpu.Buffer, n int, scalar float32) error {
	return b.elementwise(out, a, a, n, 2, scalar)
}
