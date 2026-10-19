//go:build windows && amd64

package vulkan

import (
	"fmt"
	"path/filepath"
	"syscall"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

type vkLibrary struct{ dll *syscall.DLL }

func openVulkanLibrary() (*vkLibrary, error) {
	// Load only from the system directory. kernel32 is a Windows KnownDLL.
	var system [32768]uint16
	getSystem := syscall.NewLazyDLL("kernel32.dll").NewProc("GetSystemDirectoryW")
	n, _, e := getSystem.Call(uintptr(unsafe.Pointer(&system[0])), uintptr(len(system)))
	if n == 0 || n >= uintptr(len(system)) {
		return nil, fmt.Errorf("%w: GetSystemDirectoryW: %v", gpu.ErrUnavailable, e)
	}
	dll, err := syscall.LoadDLL(filepath.Join(syscall.UTF16ToString(system[:n]), "vulkan-1.dll"))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", gpu.ErrUnavailable, err)
	}
	return &vkLibrary{dll: dll}, nil
}
func (l *vkLibrary) lookup(name string) (uintptr, error) {
	if l.dll == nil {
		return 0, gpu.ErrClosed
	}
	p, err := l.dll.FindProc(name)
	if err != nil {
		return 0, err
	}
	return p.Addr(), nil
}
func (l *vkLibrary) close() {
	if l.dll != nil {
		l.dll.Release()
		l.dll = nil
	}
}

//go:uintptrescapes
func callNative(proc uintptr, args ...uintptr) uintptr {
	r, _, _ := syscall.SyscallN(proc, args...)
	return r
}
