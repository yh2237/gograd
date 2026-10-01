//go:build windows

package cuda

import (
	"runtime"
	"unsafe"
)

// currentStream holds a stream that kernel launches use when they are not
// given one. Graph capture routes all work through a single stream this way.
var currentStream uintptr

// capturing reports whether a graph capture is in progress. Callers must avoid
// driver calls that capture forbids, such as setting the device.
var capturing bool

// InCapture reports whether a graph capture is in progress.
func InCapture() bool { return capturing }

// SetCurrentStream routes kernel launches that do not name a stream to s. Pass
// nil to use the default stream.
func SetCurrentStream(s *Stream) {
	if s == nil {
		currentStream = 0
	} else {
		currentStream = s.handle
	}
}

// Graph is a captured sequence of GPU work that replays with one launch.
type Graph struct {
	exec   uintptr
	stream *Stream
}

// Capture records the GPU work that fn enqueues on stream into a graph. fn must
// not perform synchronous copies or first-time kernel compilation: compile the
// kernels and create handles before calling. The captured work keeps referring
// to the same device buffers, so the caller must keep them allocated.
func Capture(stream *Stream, fn func() error) (*Graph, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		return nil, err
	}
	previous := currentStream
	currentStream = stream.handle
	capturing = true
	defer func() {
		currentStream = previous
		capturing = false
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
	runErr := fn()
	if code, _, _ := a.streamEndCapture.Call(stream.handle, uintptr(unsafe.Pointer(&graph))); code != 0 {
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
	return &Graph{exec: exec, stream: stream}, nil
}

// Launch replays the graph.
func (g *Graph) Launch() error {
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	if code, _, _ := a.graphLaunch.Call(g.exec, g.stream.handle); code != 0 {
		return driverError("cuGraphLaunch", code)
	}
	return nil
}

// Close releases the instantiated graph.
func (g *Graph) Close() error {
	if g.exec == 0 {
		return nil
	}
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	code, _, _ := a.graphExecDestroy.Call(g.exec)
	g.exec = 0
	return driverError("cuGraphExecDestroy", code)
}
