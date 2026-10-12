package autograd

import (
	"fmt"
	"runtime"

	"github.com/yh2237/gograd/cuda"
)

// CUDAContext pins the caller to an OS thread for CUDA driver modules. Keep it
// open through model creation, forward, backward and optimizer updates. It also
// records the selected device and restores the thread's previous device on
// Close, so an unrelated goroutine scheduled onto the same OS thread later does
// not inherit this run's device.
type CUDAContext struct {
	device   int
	previous int
	restore  bool
	closed   bool
}

// NewCUDAContext pins the calling goroutine to device 0's OS thread.
func NewCUDAContext() (*CUDAContext, error) { return NewCUDAContextOn(0) }

// NewCUDAContextOn pins the calling goroutine to an OS thread and makes device
// current on it. The device index is validated against the installed devices.
func NewCUDAContextOn(device int) (*CUDAContext, error) {
	count, err := cuda.DeviceCount()
	if err != nil {
		return nil, err
	}
	if device < 0 || device >= count {
		return nil, fmt.Errorf("autograd: CUDA device %d is out of range for %d device(s)", device, count)
	}
	previous, err := cuda.CurrentDevice()
	if err != nil {
		return nil, err
	}
	runtime.LockOSThread()
	if err := cuda.SetDevice(device); err != nil {
		runtime.UnlockOSThread()
		return nil, err
	}
	return &CUDAContext{device: device, previous: previous, restore: previous != device}, nil
}

// Device reports the device this context selected.
func (c *CUDAContext) Device() int {
	if c == nil {
		return -1
	}
	return c.device
}

// Close restores the thread's previous device and releases the OS thread.
func (c *CUDAContext) Close() {
	if c == nil || c.closed {
		return
	}
	c.closed = true
	if c.restore {
		_ = cuda.SetDevice(c.previous)
	}
	runtime.UnlockOSThread()
}
