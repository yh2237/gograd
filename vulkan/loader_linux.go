//go:build linux && !android && amd64

package vulkan

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

//go:cgo_import_dynamic x_dlopen dlopen "libdl.so.2"
//go:cgo_import_dynamic x_dlsym dlsym "libdl.so.2"
//go:cgo_import_dynamic x_dlclose dlclose "libdl.so.2"
//go:cgo_import_dynamic x_dlerror dlerror "libdl.so.2"

//go:linkname cgocall runtime.cgocall
func cgocall(fn, arg unsafe.Pointer) int32

const (
	rtldLocal = 0x00000
	rtldNow   = 0x00002
)

type sysvFrame struct {
	fn   uintptr
	args [12]uintptr
	ret  uintptr
}

func sysvCallPC() unsafe.Pointer
func dlopenAddr() uintptr
func dlsymAddr() uintptr
func dlcloseAddr() uintptr
func dlerrorAddr() uintptr
func copyCString(dst *byte, dstCap int, src uintptr) int

type vkLibrary struct{ handle uintptr }

func openVulkanLibrary() (*vkLibrary, error) { return openLinuxLibrary("libvulkan.so.1") }

func openLinuxLibrary(name string) (*vkLibrary, error) {
	cname, err := cStringBytes(name)
	if err != nil {
		return nil, fmt.Errorf("%w: dlopen(%s): %v", gpu.ErrUnavailable, name, err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	handle := callNative(dlopenAddr(), uintptr(unsafe.Pointer(&cname[0])), rtldNow|rtldLocal)
	runtime.KeepAlive(cname)
	if handle == 0 {
		return nil, fmt.Errorf("%w: dlopen(%s): %s", gpu.ErrUnavailable, name, dlErrorString())
	}
	return &vkLibrary{handle: handle}, nil
}

func (l *vkLibrary) lookup(name string) (uintptr, error) {
	if l.handle == 0 {
		return 0, gpu.ErrClosed
	}
	cname, err := cStringBytes(name)
	if err != nil {
		return 0, fmt.Errorf("dlsym(%s): %v", name, err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	callNative(dlerrorAddr())
	sym := callNative(dlsymAddr(), l.handle, uintptr(unsafe.Pointer(&cname[0])))
	runtime.KeepAlive(cname)
	if msg := dlErrorOptional(); msg != "" {
		return 0, fmt.Errorf("dlsym(%s): %s", name, msg)
	}
	if sym == 0 {
		return 0, fmt.Errorf("dlsym(%s): symbol resolved to null", name)
	}
	return sym, nil
}

func (l *vkLibrary) close() {
	if l.handle != 0 {
		callNative(dlcloseAddr(), l.handle)
		l.handle = 0
	}
}

func cStringBytes(s string) ([]byte, error) {
	for i := 0; i < len(s); i++ {
		if s[i] == 0 {
			return nil, fmt.Errorf("contains NUL byte")
		}
	}
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b, nil
}

func dlErrorOptional() string {
	p := callNative(dlerrorAddr())
	if p == 0 {
		return ""
	}
	var buf [512]byte
	n := copyCString(&buf[0], len(buf), p)
	return string(buf[:n])
}

func dlErrorString() string {
	if msg := dlErrorOptional(); msg != "" {
		return msg
	}
	return "unknown dl error"
}

//go:uintptrescapes
func callNative(proc uintptr, args ...uintptr) uintptr {
	if len(args) > 12 {
		panic("vulkan: too many native call arguments")
	}
	var f sysvFrame
	f.fn = proc
	copy(f.args[:], args)
	cgocall(sysvCallPC(), unsafe.Pointer(&f))
	runtime.KeepAlive(args)
	return f.ret
}
