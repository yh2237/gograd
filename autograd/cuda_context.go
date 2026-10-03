package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"runtime"
)

// CUDAContext pins the caller to an OS thread for CUDA driver modules. Keep it
// open through model creation, forward, backward and optimizer updates.
type CUDAContext struct{ closed bool }

func NewCUDAContext() (*CUDAContext, error) {
	runtime.LockOSThread()
	if e := cuda.SetDevice(0); e != nil {
		runtime.UnlockOSThread()
		return nil, e
	}
	return &CUDAContext{}, nil
}
func (c *CUDAContext) Close() {
	if c != nil && !c.closed {
		c.closed = true
		runtime.UnlockOSThread()
	}
}
