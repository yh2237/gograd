//go:build windows

package cuda

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// CUDA's notion of a current device and a current stream is per host thread, and
// this package follows the same rule instead of sharing one process-wide setting.
// A goroutine that uses CUDA locks its OS thread (see NewCUDAContext), so
// keying this state by the OS thread id keeps independent training runs from
// overwriting each other's stream or capture state.
var currentThreadIDProc = syscall.NewLazyDLL("kernel32.dll").NewProc("GetCurrentThreadId")

func currentThreadID() uintptr {
	id, _, _ := currentThreadIDProc.Call()
	return id
}

// ThreadID identifies the calling OS thread. CUDA scopes its current device,
// current stream and capture state per host thread, so a caller that keeps
// thread-owned driver objects can key them by this value.
func ThreadID() uintptr { return currentThreadID() }

// launchState is the implicit CUDA state of one OS thread.
type launchState struct {
	stream    uintptr
	streamRef *Stream
	capturing bool
	// touched marks pool pointers used while this thread captures a graph. It
	// stays per thread so two concurrent captures cannot clobber each other.
	touched map[uintptr]int
}

var launchStates sync.Map // uintptr -> *launchState

func launchStateForThread() *launchState {
	id := currentThreadID()
	if value, ok := launchStates.Load(id); ok {
		return value.(*launchState)
	}
	state := &launchState{}
	actual, _ := launchStates.LoadOrStore(id, state)
	return actual.(*launchState)
}

// releaseIdleThreadState drops the entry for a thread with no active stream or
// capture. Go reuses OS threads, so leaving stale entries would grow the map for
// the life of the process.
func (s *launchState) releaseIfIdle() {
	if s.stream != 0 || s.capturing {
		return
	}
	launchStates.CompareAndDelete(currentThreadID(), s)
}

// currentStreamHandle returns the stream implicit launches should use on this
// thread. Zero selects the default stream.
func currentStreamHandle() uintptr {
	return launchStateForThread().stream
}

// InCapture reports whether a graph capture is in progress on this thread.
// Driver calls that capture forbids, such as setting the device, must be avoided
// while it is true.
func InCapture() bool { return launchStateForThread().capturing }

// SetCurrentStream routes kernel launches that do not name a stream to s on the
// calling OS thread. Pass nil to use the default stream.
func SetCurrentStream(s *Stream) {
	state := launchStateForThread()
	if s == nil {
		state.stream, state.streamRef = 0, nil
		state.releaseIfIdle()
		return
	}
	state.stream, state.streamRef = s.handle, s
}

// CurrentStream returns the stream selected for the calling OS thread, or nil
// for the default stream. Callers that bind thread-owned driver objects, such as
// a cuBLAS handle, use it to follow the implicit launch stream.
func CurrentStream() *Stream { return launchStateForThread().streamRef }

// WithStream runs fn with stream selected as the calling OS thread's implicit
// stream and restores the previous selection afterwards, including when fn
// panics. Kernel launches, cuBLAS calls that follow the implicit stream, and
// buffer operations inside fn use stream.
func WithStream(stream *Stream, fn func() error) error {
	if fn == nil {
		return fmt.Errorf("cuda: nil stream callback")
	}
	previous := CurrentStream()
	SetCurrentStream(stream)
	defer SetCurrentStream(previous)
	return fn()
}

// CurrentStreamHandle reports the stream handle implicit launches use on this
// thread. Zero means the default stream.
func CurrentStreamHandle() uintptr { return currentStreamHandle() }

// Graph is a captured sequence of GPU work that replays with one launch.
type Graph struct {
	exec     uintptr
	stream   *Stream
	device   int
	reserved map[uintptr]int
}

// Capture records the GPU work that fn enqueues on stream into a graph. fn must
// not perform synchronous copies or first-time kernel compilation: compile the
// kernels and create handles before calling. The captured work keeps referring
// to the same device buffers, so the caller must keep them allocated.
//
// Capture owns the calling OS thread for its duration, records the current
// device without changing it, and rejects a nested capture on the same thread
// instead of silently corrupting the outer one.
func Capture(stream *Stream, fn func() error) (*Graph, error) {
	if stream == nil {
		return nil, fmt.Errorf("cuda: capture requires an explicit stream")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	state := launchStateForThread()
	if state.capturing {
		return nil, fmt.Errorf("cuda: nested graph capture on one OS thread is not supported")
	}
	device, err := CurrentDevice()
	if err != nil {
		return nil, err
	}
	previous, previousRef := state.stream, state.streamRef
	state.stream, state.streamRef = stream.handle, stream
	state.capturing = true
	poolMu.Lock()
	state.touched = map[uintptr]int{}
	poolMu.Unlock()
	defer func() {
		state.stream, state.streamRef = previous, previousRef
		state.capturing = false
		poolMu.Lock()
		state.touched = nil
		poolMu.Unlock()
		state.releaseIfIdle()
	}()
	a, err := loadKernelAPI()
	if err != nil {
		return nil, err
	}
	// CU_STREAM_CAPTURE_MODE_RELAXED allows cuBLAS to use its own internal
	// streams during capture.
	if code, _, _ := a.streamBeginCapture.Call(stream.handle, 2); code != 0 {
		return nil, driverError("cuStreamBeginCapture", code)
	}
	var graph uintptr
	captureEnded := false
	defer func() {
		if captureEnded {
			return
		}
		// The callback panicked. End the capture anyway, otherwise the stream
		// stays in capture mode and rejects every later operation.
		if code, _, _ := a.streamEndCapture.Call(stream.handle, uintptr(unsafe.Pointer(&graph))); code != 0 && graph != 0 {
			a.graphDestroy.Call(graph)
		}
	}()
	runErr := fn()
	code, _, _ := a.streamEndCapture.Call(stream.handle, uintptr(unsafe.Pointer(&graph)))
	captureEnded = true
	if code != 0 {
		if graph != 0 {
			a.graphDestroy.Call(graph)
		}
		return nil, driverError("cuStreamEndCapture", code)
	}
	if runErr != nil {
		if graph != 0 {
			a.graphDestroy.Call(graph)
		}
		return nil, runErr
	}
	var exec uintptr
	if code, _, _ := a.graphInstantiate.Call(uintptr(unsafe.Pointer(&exec)), graph, 0); code != 0 {
		a.graphDestroy.Call(graph)
		return nil, driverError("cuGraphInstantiate", code)
	}
	a.graphDestroy.Call(graph)
	reserved := map[uintptr]int{}
	poolMu.Lock()
	for key, list := range poolTable {
		kept := list[:0]
		for _, pointer := range list {
			if _, used := state.touched[pointer]; used {
				reserved[pointer] = key.size
				uncacheAccountingLocked(pointer, key.size)
			} else {
				kept = append(kept, pointer)
			}
		}
		poolTable[key] = kept
	}
	poolMu.Unlock()
	return &Graph{exec: exec, stream: stream, device: device, reserved: reserved}, nil
}

// Launch replays the graph on the device it was captured on.
func (g *Graph) Launch() error {
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	return withDevice(g.device, func() error {
		code, _, _ := a.graphLaunch.Call(g.exec, g.stream.handle)
		return driverError("cuGraphLaunch", code)
	})
}

// Close releases the instantiated graph and returns its reserved allocations to
// the pool on the device they belong to.
func (g *Graph) Close() error {
	if g.exec == 0 {
		return nil
	}
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	if err := g.stream.Synchronize(); err != nil {
		return err
	}
	code, _, _ := a.graphExecDestroy.Call(g.exec)
	g.exec = 0
	poolMu.Lock()
	for pointer, size := range g.reserved {
		cachePointerLocked(pointer, size, g.device, g.stream.handle)
	}
	g.reserved = nil
	victims := trimPoolLocked()
	poolMu.Unlock()
	if err := driverError("cuGraphExecDestroy", code); err != nil {
		return err
	}
	return freeCachedPointers(victims)
}
