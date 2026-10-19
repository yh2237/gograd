package cuda

import (
	"errors"
	"runtime"
	"testing"
	"unsafe"
)

const auditDeviceSource = `extern "C" __global__ void write_value(float* out){if(threadIdx.x==0)out[0]=7.0f;}`

func TestAuditCompilePreservesCurrentDevice(t *testing.T) {
	if !Available() {
		t.Skip("CUDA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	previous, err := CurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer SetDevice(previous)
	count, err := DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count == 0 {
		t.Skip("no CUDA devices")
	}
	// Also runs on one GPU; on multi-GPU hosts this specifically compiles on
	// a nonzero current device before any buffer initializes its context.
	device := count - 1
	if err := SetDevice(device); err != nil {
		t.Fatal(err)
	}
	program, err := Compile(auditDeviceSource)
	if err != nil {
		t.Fatal(err)
	}
	defer program.Close()
	current, err := CurrentDevice()
	if err != nil || current != device {
		t.Fatalf("Compile changed current device %d to %d: %v", device, current, err)
	}
	function, err := program.Function("write_value")
	if err != nil {
		t.Fatal(err)
	}
	buffer, err := Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Free()
	p := buffer.Pointer()
	if err := function.Launch([3]int{1, 1, 1}, [3]int{32, 1, 1}, 0, nil, []unsafe.Pointer{unsafe.Pointer(&p)}); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	if got := download(t, buffer, 1)[0]; got != 7 {
		t.Fatalf("kernel on device %d returned %g", device, got)
	}
}

func TestAuditKernelCacheMultipleDevices(t *testing.T) {
	if !Available() {
		t.Skip("CUDA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	count, err := DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count < 2 {
		t.Skip("requires two CUDA devices")
	}
	previous, err := CurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer SetDevice(previous)
	var cache KernelCache
	seen := make(map[int]*Kernel)
	defer func() {
		for device, kernel := range seen {
			_ = SetDevice(device)
			program := &Program{module: kernel.module}
			_ = program.Close()
		}
	}()
	for _, device := range []int{0, 1, 0, 1} {
		if err := SetDevice(device); err != nil {
			t.Fatal(err)
		}
		functions, err := cache.Get(auditDeviceSource, "write_value")
		if err != nil {
			t.Fatal(err)
		}
		kernel := functions["write_value"]
		if old := seen[device]; old != nil && old != kernel {
			t.Fatal("kernel cache failed to reuse current device entry")
		}
		if old := seen[1-device]; old == kernel {
			t.Fatal("kernel cache reused a function from the other device")
		}
		seen[device] = kernel
		buffer, err := Alloc(4)
		if err != nil {
			t.Fatal(err)
		}
		p := buffer.Pointer()
		if err := kernel.Launch([3]int{1, 1, 1}, [3]int{32, 1, 1}, 0, nil, []unsafe.Pointer{unsafe.Pointer(&p)}); err != nil {
			t.Fatal(err)
		}
		if err := Synchronize(); err != nil {
			t.Fatal(err)
		}
		if got := download(t, buffer, 1)[0]; got != 7 {
			t.Fatalf("device %d value %g, want 7", device, got)
		}
		buffer.Free()
	}
}

func TestAuditDeviceCacheKeepsErrors(t *testing.T) {
	if !Available() {
		t.Skip("CUDA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var cache DeviceCache[int]
	want := errors.New("test build failure")
	calls := 0
	build := func() (int, error) { calls++; return 0, want }
	for range 2 {
		if _, err := cache.Get(build); err != want {
			t.Fatalf("cached error %v, want %v", err, want)
		}
	}
	if calls != 1 {
		t.Fatalf("failed builder called %d times, want 1", calls)
	}
}
