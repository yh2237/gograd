//go:build windows && amd64

package directml

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

type guid struct {
	A    uint32
	B, C uint16
	D    [8]byte
}

func iid(s string) *guid {
	v, err := hex.DecodeString(strings.ReplaceAll(s, "-", ""))
	if err != nil || len(v) != 16 {
		panic("invalid internal GUID")
	}
	g := &guid{A: binary.BigEndian.Uint32(v), B: binary.BigEndian.Uint16(v[4:]), C: binary.BigEndian.Uint16(v[6:])}
	copy(g.D[:], v[8:])
	return g
}

// object points to native COM memory, not a Go-owned vtable or wrapper.
type object struct{ vtable *[64]uintptr }

//go:uintptrescapes
func call(o *object, index int, args ...uintptr) uintptr {
	r, _, _ := syscall.SyscallN(o.vtable[index], append([]uintptr{uintptr(unsafe.Pointer(o))}, args...)...)
	return r
}
func release(o *object) {
	if o != nil {
		call(o, 2)
	}
}
func hresult(op string, r uintptr) error {
	if int32(r) < 0 {
		return fmt.Errorf("directml: %s HRESULT 0x%08x", op, uint32(r))
	}
	return nil
}

func systemDLL(name string) (*syscall.DLL, error) {
	var buf [32768]uint16
	getSystem := syscall.NewLazyDLL("kernel32.dll").NewProc("GetSystemDirectoryW")
	n, _, err := getSystem.Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if n == 0 || n >= uintptr(len(buf)) {
		return nil, fmt.Errorf("GetSystemDirectoryW: %v", err)
	}
	return syscall.LoadDLL(filepath.Join(syscall.UTF16ToString(buf[:n]), name))
}

var (
	iidFactory      = iid("770aae78-f26f-4dba-a829-253c83d1b387")
	iidD3DDevice    = iid("189819f1-1db6-4b57-be54-1821339b85f7")
	iidQueue        = iid("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
	iidAllocator    = iid("6102dee4-af59-4b09-b999-b44d73f09b24")
	iidCommand      = iid("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455")
	iidFence        = iid("0a753dcf-c4d8-4b91-adf6-be5a60d95a76")
	iidResource     = iid("696442be-a72e-4059-bc79-5b5c98040fad")
	iidHeap         = iid("8efb471d-616c-4f49-90f7-127bb763fa51")
	iidDMLDevice    = iid("6dbd6437-96fd-423f-a98c-ae5e7c2a573f")
	iidOperator     = iid("26caae7a-3081-4633-9581-226fbe57695d")
	iidCompiled     = iid("6b15e56a-bf5c-4902-92d8-da3a650afea4")
	iidInitializer  = iid("427c1113-435c-469c-8676-4d5dd072f813")
	iidBindingTable = iid("29c687dc-de74-4e3b-ab00-1168f2fc3cfc")
	iidRecorder     = iid("e6857a76-2e3e-4fdd-bff4-5d2ba10fb453")
)
