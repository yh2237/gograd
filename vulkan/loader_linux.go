//go:build linux && !android && amd64

package vulkan

import (
	"fmt"

	"github.com/ebitengine/purego"
	"github.com/yh2237/gograd/gpu"
)

type vkLibrary struct{ handle uintptr }

func openVulkanLibrary() (*vkLibrary, error) { return openLinuxLibrary("libvulkan.so.1") }

func openLinuxLibrary(name string) (*vkLibrary, error) {
	// Use the system dynamic linker's normal SONAME resolution. Do not try a
	// local ./libvulkan.so or an unversioned development-library fallback.
	handle, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("%w: dlopen(%s): %v", gpu.ErrUnavailable, name, err)
	}
	return &vkLibrary{handle: handle}, nil
}
func (l *vkLibrary) lookup(name string) (uintptr, error) {
	if l.handle == 0 {
		return 0, gpu.ErrClosed
	}
	return purego.Dlsym(l.handle, name)
}
func (l *vkLibrary) close() {
	if l.handle != 0 {
		purego.Dlclose(l.handle)
		l.handle = 0
	}
}

// Vulkan's functions used here have integer/pointer arguments only: floating
// point values reside in pointed-to structs, not SysV XMM argument registers.
// purego handles stack arguments, native stack switching and scheduler entry.
// uintptrescapes keeps Go arguments on the heap and live through the call.
//
//go:uintptrescapes
func callNative(proc uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(proc, args...)
	return r
}
