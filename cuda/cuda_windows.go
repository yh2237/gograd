//go:build windows

// Package cuda binds the CUDA runtime and cuBLAS DLLs through the Windows
// loader. No C compiler or cgo is required: the DLLs are resolved lazily and
// their C entry points are called directly.
package cuda

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"
)

// ErrUnavailable reports that the CUDA runtime or cuBLAS DLL could not be
// loaded.
var ErrUnavailable = errors.New("cuda: runtime is not available")

// Error is a CUDA or cuBLAS failure with the API name and status text.
type Error struct {
	Op      string
	Code    int
	Message string
}

func (e *Error) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("cuda: %s failed (code %d)", e.Op, e.Code)
	}
	return fmt.Sprintf("cuda: %s failed (code %d): %s", e.Op, e.Code, e.Message)
}

type api struct {
	cudart *syscall.DLL
	cublas *syscall.DLL

	getDeviceCount     *syscall.Proc
	getDevice          *syscall.Proc
	setDevice          *syscall.Proc
	deviceSynchronize  *syscall.Proc
	getLastError       *syscall.Proc
	getDeviceAttribute *syscall.Proc
	malloc             *syscall.Proc
	free               *syscall.Proc
	memcpy             *syscall.Proc
	memset             *syscall.Proc
	streamCreate       *syscall.Proc
	streamDestroy      *syscall.Proc
	streamSynchronize  *syscall.Proc
	streamWaitEvent    *syscall.Proc
	eventCreate        *syscall.Proc
	eventDestroy       *syscall.Proc
	eventRecord        *syscall.Proc
	eventSynchronize   *syscall.Proc

	cublasCreate       *syscall.Proc
	cublasDestroy      *syscall.Proc
	cublasSetStream    *syscall.Proc
	cublasSgemm        *syscall.Proc
	cublasStridedBatch *syscall.Proc
}

var (
	loadOnce sync.Once
	loaded   *api
	loadErr  error
)

func libraryCandidates(base string) []string {
	names := []string{base}
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		names = append(names,
			filepath.Join(cudaPath, "bin", "x64", base),
			filepath.Join(cudaPath, "bin", base),
		)
	}
	return names
}

func loadLibrary(names []string) (*syscall.DLL, error) {
	var lastErr error
	for _, name := range names {
		dll, err := syscall.LoadDLL(name)
		if err == nil {
			return dll, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

func findProcs(dll *syscall.DLL, names ...string) ([]*syscall.Proc, error) {
	procs := make([]*syscall.Proc, len(names))
	for i, name := range names {
		proc, err := dll.FindProc(name)
		if err != nil {
			return nil, err
		}
		procs[i] = proc
	}
	return procs, nil
}

// findProcsAny resolves each entry to the first available of its alternative
// symbol names, which handles APIs that changed suffix across CUDA versions.
func findProcsAny(dll *syscall.DLL, alternatives [][]string) ([]*syscall.Proc, error) {
	procs := make([]*syscall.Proc, len(alternatives))
	for i, names := range alternatives {
		var found *syscall.Proc
		var lastErr error
		for _, name := range names {
			proc, err := dll.FindProc(name)
			if err == nil {
				found = proc
				break
			}
			lastErr = err
		}
		if found == nil {
			return nil, lastErr
		}
		procs[i] = found
	}
	return procs, nil
}

func load() (*api, error) {
	loadOnce.Do(func() {
		runtimeNames := libraryCandidates("cudart64_13.dll")
		runtimeNames = append(runtimeNames,
			libraryCandidates("cudart64_12.dll")...,
		)
		runtimeNames = append(runtimeNames,
			libraryCandidates("cudart64_11.dll")...,
		)
		cudart, err := loadLibrary(runtimeNames)
		if err != nil {
			loadErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		blasNames := libraryCandidates("cublas64_13.dll")
		blasNames = append(blasNames, libraryCandidates("cublas64_12.dll")...)
		blasNames = append(blasNames, libraryCandidates("cublas64_11.dll")...)
		cublas, err := loadLibrary(blasNames)
		if err != nil {
			loadErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}

		runtime, err := findProcs(cudart,
			"cudaGetDeviceCount", "cudaGetDevice", "cudaSetDevice",
			"cudaDeviceSynchronize", "cudaGetLastError", "cudaDeviceGetAttribute",
			"cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemset",
			"cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize",
			"cudaStreamWaitEvent", "cudaEventCreate", "cudaEventDestroy",
			"cudaEventRecord", "cudaEventSynchronize",
		)
		if err != nil {
			loadErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		blas, err := findProcs(cublas,
			"cublasCreate_v2", "cublasDestroy_v2", "cublasSetStream_v2", "cublasSgemm_v2",
			"cublasSgemmStridedBatched",
		)
		if err != nil {
			loadErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		loaded = &api{
			cudart: cudart, cublas: cublas,
			getDeviceCount: runtime[0], getDevice: runtime[1], setDevice: runtime[2],
			deviceSynchronize: runtime[3], getLastError: runtime[4], getDeviceAttribute: runtime[5],
			malloc: runtime[6], free: runtime[7], memcpy: runtime[8], memset: runtime[9],
			streamCreate: runtime[10], streamDestroy: runtime[11], streamSynchronize: runtime[12],
			streamWaitEvent: runtime[13], eventCreate: runtime[14], eventDestroy: runtime[15],
			eventRecord: runtime[16], eventSynchronize: runtime[17],
			cublasCreate: blas[0], cublasDestroy: blas[1], cublasSetStream: blas[2],
			cublasSgemm: blas[3], cublasStridedBatch: blas[4],
		}
	})
	return loaded, loadErr
}

// Available reports whether the CUDA runtime and cuBLAS DLLs loaded.
func Available() bool {
	_, err := load()
	return err == nil
}

// runtimeMessages names the runtime errors most likely to surface here. The
// numeric code is always reported; unknown codes have no text.
var runtimeMessages = map[int]string{
	1:   "invalid value",
	2:   "out of memory",
	3:   "initialization error",
	4:   "dll not found",
	100: "no CUDA-capable device is detected",
	101: "invalid device ordinal",
	201: "invalid device pointer",
	209: "no kernel image is available for execution on the device",
	700: "illegal memory access",
	701: "launch out of resources",
	719: "unspecified launch failure",
}

func runtimeError(op string, code uintptr) error {
	if code == 0 {
		return nil
	}
	return &Error{Op: op, Code: int(code), Message: runtimeMessages[int(code)]}
}

func blasError(op string, status uintptr) error {
	if status == 0 {
		return nil
	}
	return &Error{Op: op, Code: int(status)}
}

// DeviceCount returns the number of CUDA devices.
func DeviceCount() (int, error) {
	a, err := load()
	if err != nil {
		return 0, err
	}
	var count int32
	code, _, _ := a.getDeviceCount.Call(uintptr(unsafe.Pointer(&count)))
	if err := runtimeError("cudaGetDeviceCount", code); err != nil {
		return 0, err
	}
	return int(count), nil
}

// SetDevice makes the given device current on the calling goroutine.
func SetDevice(device int) error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.setDevice.Call(uintptr(int32(device)))
	return runtimeError("cudaSetDevice", code)
}

// CurrentDevice returns the current device index.
func CurrentDevice() (int, error) {
	a, err := load()
	if err != nil {
		return 0, err
	}
	var device int32
	code, _, _ := a.getDevice.Call(uintptr(unsafe.Pointer(&device)))
	if err := runtimeError("cudaGetDevice", code); err != nil {
		return 0, err
	}
	return int(device), nil
}

// DeviceAttribute returns an integer device attribute. Attribute 75 is the
// compute capability major version and 76 is the minor version.
func DeviceAttribute(attribute, device int) (int, error) {
	a, err := load()
	if err != nil {
		return 0, err
	}
	var value int32
	code, _, _ := a.getDeviceAttribute.Call(uintptr(unsafe.Pointer(&value)), uintptr(int32(attribute)), uintptr(int32(device)))
	if err := runtimeError("cudaDeviceGetAttribute", code); err != nil {
		return 0, err
	}
	return int(value), nil
}

// initializeContext creates the primary context on the current device via the
// documented cudaFree(0) idiom.
func initializeContext() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.free.Call(0)
	return runtimeError("cudaFree", code)
}

// Synchronize blocks until all preceding work on the current device finishes
// and clears the sticky error state.
func Synchronize() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.deviceSynchronize.Call()
	return runtimeError("cudaDeviceSynchronize", code)
}

// LastError returns the last asynchronous error recorded by the runtime.
func LastError() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.getLastError.Call()
	return runtimeError("cudaGetLastError", code)
}

// Buffer is a device allocation. Callers must Free it explicitly; no finalizer
// is used because release order matters.
type Buffer struct {
	pointer uintptr
	size    int
	freed   bool
}

var (
	poolMu    sync.Mutex
	poolTable = map[int][]uintptr{}
)

// Alloc reserves size bytes on the current device. Freed buffers of the same
// size are reused from an internal pool, which avoids the cost of repeated
// cudaMalloc/cudaFree pairs.
func Alloc(size int) (*Buffer, error) {
	if size < 0 {
		return nil, fmt.Errorf("cuda: negative allocation size %d", size)
	}
	a, err := load()
	if err != nil {
		return nil, err
	}
	poolMu.Lock()
	list := poolTable[size]
	if len(list) > 0 {
		pointer := list[len(list)-1]
		poolTable[size] = list[:len(list)-1]
		poolMu.Unlock()
		return &Buffer{pointer: pointer, size: size}, nil
	}
	poolMu.Unlock()
	var pointer uintptr
	code, _, _ := a.malloc.Call(uintptr(unsafe.Pointer(&pointer)), uintptr(size))
	if err := runtimeError("cudaMalloc", code); err != nil {
		return nil, err
	}
	return &Buffer{pointer: pointer, size: size}, nil
}

// Free returns the allocation to the internal pool. It is safe to call more
// than once. Use ReleasePool to release pooled memory to the driver.
func (b *Buffer) Free() error {
	if b == nil || b.freed {
		return nil
	}
	b.freed = true
	poolMu.Lock()
	poolTable[b.size] = append(poolTable[b.size], b.pointer)
	poolMu.Unlock()
	b.pointer = 0
	return nil
}

// ReleasePool frees every buffer currently held by the internal pool.
func ReleasePool() error {
	a, err := load()
	if err != nil {
		return err
	}
	poolMu.Lock()
	defer poolMu.Unlock()
	for _, list := range poolTable {
		for _, pointer := range list {
			code, _, _ := a.free.Call(pointer)
			if err := runtimeError("cudaFree", code); err != nil {
				return err
			}
		}
	}
	poolTable = map[int][]uintptr{}
	return nil
}

// Pointer returns the raw device address.
func (b *Buffer) Pointer() uintptr { return b.pointer }

// Size returns the allocation size in bytes.
func (b *Buffer) Size() int { return b.size }

// CopyFromHost uploads len(data) bytes to the start of the buffer.
func (b *Buffer) CopyFromHost(data []byte) error {
	return b.copy(1, data, 0)
}

// CopyToHost downloads len(data) bytes from the start of the buffer.
func (b *Buffer) CopyToHost(data []byte) error {
	return b.copy(2, data, 0)
}

func (b *Buffer) copy(kind int, data []byte, offset int) error {
	if offset < 0 || offset+len(data) > b.size {
		return fmt.Errorf("cuda: copy range [%d,%d) exceeds buffer size %d", offset, offset+len(data), b.size)
	}
	if len(data) == 0 {
		return nil
	}
	a, err := load()
	if err != nil {
		return err
	}
	var source uintptr
	if kind == 1 {
		source = uintptr(unsafe.Pointer(&data[0]))
	}
	destination := b.pointer + uintptr(offset)
	if kind == 2 {
		destination, source = uintptr(unsafe.Pointer(&data[0])), b.pointer+uintptr(offset)
	}
	code, _, _ := a.memcpy.Call(destination, source, uintptr(len(data)), uintptr(kind))
	return runtimeError("cudaMemcpy", code)
}

// Memset fills the first size bytes with a byte value.
func (b *Buffer) Memset(value byte, size int) error {
	if size < 0 || size > b.size {
		return fmt.Errorf("cuda: memset size %d exceeds buffer size %d", size, b.size)
	}
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.memset.Call(b.pointer, uintptr(value), uintptr(size))
	return runtimeError("cudaMemset", code)
}

// Stream is a non-default CUDA stream.
type Stream struct {
	handle uintptr
}

// NewStream creates a stream.
func NewStream() (*Stream, error) {
	a, err := load()
	if err != nil {
		return nil, err
	}
	var handle uintptr
	code, _, _ := a.streamCreate.Call(uintptr(unsafe.Pointer(&handle)))
	if err := runtimeError("cudaStreamCreate", code); err != nil {
		return nil, err
	}
	return &Stream{handle: handle}, nil
}

// Synchronize waits for all work on the stream.
func (s *Stream) Synchronize() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.streamSynchronize.Call(s.handle)
	return runtimeError("cudaStreamSynchronize", code)
}

// Destroy releases the stream.
func (s *Stream) Destroy() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.streamDestroy.Call(s.handle)
	s.handle = 0
	return runtimeError("cudaStreamDestroy", code)
}

// Event is a CUDA event used for ordering and timing.
type Event struct {
	handle uintptr
}

// NewEvent creates an event with the given timing flag.
func NewEvent(timing bool) (*Event, error) {
	a, err := load()
	if err != nil {
		return nil, err
	}
	flags := uintptr(0)
	if timing {
		flags = 2 // cudaEventDefault is enough to observe completion
	}
	var handle uintptr
	code, _, _ := a.eventCreate.Call(uintptr(unsafe.Pointer(&handle)), flags)
	if err := runtimeError("cudaEventCreate", code); err != nil {
		return nil, err
	}
	return &Event{handle: handle}, nil
}

// Record marks the event on the stream.
func (e *Event) Record(stream *Stream) error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.eventRecord.Call(e.handle, stream.handle)
	return runtimeError("cudaEventRecord", code)
}

// Synchronize waits for the event.
func (e *Event) Synchronize() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.eventSynchronize.Call(e.handle)
	return runtimeError("cudaEventSynchronize", code)
}

// Destroy releases the event.
func (e *Event) Destroy() error {
	a, err := load()
	if err != nil {
		return err
	}
	code, _, _ := a.eventDestroy.Call(e.handle)
	e.handle = 0
	return runtimeError("cudaEventDestroy", code)
}

// Blas is a cuBLAS handle. Operations are issued in column-major order, the
// native cuBLAS convention; use SgemmRowMajor for row-major inputs.
type Blas struct {
	handle uintptr
}

// NewBlas creates a cuBLAS handle.
func NewBlas() (*Blas, error) {
	a, err := load()
	if err != nil {
		return nil, err
	}
	var handle uintptr
	status, _, _ := a.cublasCreate.Call(uintptr(unsafe.Pointer(&handle)))
	if err := blasError("cublasCreate", status); err != nil {
		return nil, err
	}
	return &Blas{handle: handle}, nil
}

// SetStream binds the handle to a stream.
func (b *Blas) SetStream(stream *Stream) error {
	a, err := load()
	if err != nil {
		return err
	}
	status, _, _ := a.cublasSetStream.Call(b.handle, stream.handle)
	return blasError("cublasSetStream", status)
}

// Destroy releases the handle.
func (b *Blas) Destroy() error {
	a, err := load()
	if err != nil {
		return err
	}
	status, _, _ := a.cublasDestroy.Call(b.handle)
	b.handle = 0
	return blasError("cublasDestroy", status)
}

// SgemmRowMajorNT computes c = alpha*a*b^T + beta*c for row-major float32
// matrices a [m,k], b [n,k] and c [m,n]. Leading dimensions must be at least
// the corresponding row width.
func (b *Blas) SgemmRowMajorNT(m, n, k int, alpha float32, a uintptr, lda int, bPtr uintptr, ldb int, beta float32, c uintptr, ldc int) error {
	aAPI, err := load()
	if err != nil {
		return err
	}
	// Column-major cublasSgemm with the weight operand transposed produces the
	// row-major product a * b^T.
	status, _, _ := aAPI.cublasSgemm.Call(
		b.handle,
		1, 0, // CUBLAS_OP_T, CUBLAS_OP_N
		uintptr(int32(n)), uintptr(int32(m)), uintptr(int32(k)),
		uintptr(unsafe.Pointer(&alpha)),
		bPtr, uintptr(int32(ldb)),
		a, uintptr(int32(lda)),
		uintptr(unsafe.Pointer(&beta)),
		c, uintptr(int32(ldc)),
	)
	return blasError("cublasSgemm", status)
}

// SgemmStridedBatchedRowMajor computes c = alpha*a*b + beta*c for batchCount
// row-major float32 matrix products a [m,k] (stride strideA), b [k,n] (stride
// strideB) and c [m,n] (stride strideC). A zero stride repeats one operand
// across the batch.
func (b *Blas) SgemmStridedBatchedRowMajor(batchCount, m, n, k int, alpha float32, a uintptr, lda int, strideA int64, bPtr uintptr, ldb int, strideB int64, beta float32, c uintptr, ldc int, strideC int64) error {
	aAPI, err := load()
	if err != nil {
		return err
	}
	status, _, _ := aAPI.cublasStridedBatch.Call(
		b.handle,
		0, 0, // CUBLAS_OP_N, CUBLAS_OP_N
		uintptr(int32(n)), uintptr(int32(m)), uintptr(int32(k)),
		uintptr(unsafe.Pointer(&alpha)),
		bPtr, uintptr(int32(ldb)), uintptr(strideB),
		a, uintptr(int32(lda)), uintptr(strideA),
		uintptr(unsafe.Pointer(&beta)),
		c, uintptr(int32(ldc)), uintptr(strideC),
		uintptr(int32(batchCount)),
	)
	return blasError("cublasSgemmStridedBatched", status)
}

// SgemmRowMajorTransposeA computes c = alpha*a^T*b + beta*c for row-major
// float32 matrices a [k,m], b [k,n] and c [m,n]. Leading dimensions must be at
// least the corresponding row width.
func (b *Blas) SgemmRowMajorTransposeA(m, n, k int, alpha float32, a uintptr, lda int, bPtr uintptr, ldb int, beta float32, c uintptr, ldc int) error {
	aAPI, err := load()
	if err != nil {
		return err
	}
	status, _, _ := aAPI.cublasSgemm.Call(
		b.handle,
		0, 1, // CUBLAS_OP_N, CUBLAS_OP_T
		uintptr(int32(n)), uintptr(int32(m)), uintptr(int32(k)),
		uintptr(unsafe.Pointer(&alpha)),
		bPtr, uintptr(int32(ldb)),
		a, uintptr(int32(lda)),
		uintptr(unsafe.Pointer(&beta)),
		c, uintptr(int32(ldc)),
	)
	return blasError("cublasSgemm", status)
}

// SgemmRowMajor computes c = alpha*a*b + beta*c for row-major float32 matrices
// a [m,k], b [k,n] and c [m,n]. Leading dimensions must be at least the
// corresponding row width.
func (b *Blas) SgemmRowMajor(m, n, k int, alpha float32, a uintptr, lda int, bPtr uintptr, ldb int, beta float32, c uintptr, ldc int) error {
	aAPI, err := load()
	if err != nil {
		return err
	}
	// cublasSgemm is column-major. Passing the operands swapped computes the
	// row-major product with the same memory.
	status, _, _ := aAPI.cublasSgemm.Call(
		b.handle,
		0, 0, // CUBLAS_OP_N
		uintptr(int32(n)), uintptr(int32(m)), uintptr(int32(k)),
		uintptr(unsafe.Pointer(&alpha)),
		bPtr, uintptr(int32(ldb)),
		a, uintptr(int32(lda)),
		uintptr(unsafe.Pointer(&beta)),
		c, uintptr(int32(ldc)),
	)
	return blasError("cublasSgemm", status)
}
