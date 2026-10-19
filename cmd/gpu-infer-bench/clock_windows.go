//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

// Explicit QPC timing avoids sub-millisecond quantization observed with the
// Windows Go wall/monotonic clock on this host. Timer resolution is not changed.
var counterProc = syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceCounter")
var counterFrequency = func() int64 {
	var frequency int64
	r, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceFrequency").Call(uintptr(unsafe.Pointer(&frequency)))
	if r == 0 || frequency <= 0 {
		panic("QueryPerformanceFrequency unavailable")
	}
	return frequency
}()

func clockNow() int64 {
	var value int64
	r, _, _ := counterProc.Call(uintptr(unsafe.Pointer(&value)))
	if r == 0 {
		panic("QueryPerformanceCounter failed")
	}
	return value
}
func elapsedMS(start int64) float64 {
	return float64(clockNow()-start) * 1000 / float64(counterFrequency)
}
