package autograd

import (
	"testing"

	"github.com/yh2237/gograd/cuda"
)

func TestCUDAContextDeviceSelection(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	count, err := cuda.DeviceCount()
	if err != nil || count < 1 {
		t.Fatalf("device count %d %v", count, err)
	}
	context, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	if context.Device() != 0 {
		t.Fatalf("default context device %d", context.Device())
	}
	device, err := cuda.CurrentDevice()
	if err != nil || device != 0 {
		t.Fatalf("current device %d %v", device, err)
	}
	context.Close()
	context.Close() // closing twice is safe
	if device, err := cuda.CurrentDevice(); err != nil || device != 0 {
		t.Fatalf("device after close %d %v", device, err)
	}
}

func TestCUDAContextValidatesDevice(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	count, err := cuda.DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	for name, index := range map[string]int{"negative": -1, "out of range": count} {
		if _, err := NewCUDAContextOn(index); err == nil {
			t.Fatalf("%s device index accepted", name)
		}
	}
	// A rejected index must not leave the OS thread locked: a later valid
	// context still obtains and releases its own thread.
	context, err := NewCUDAContextOn(0)
	if err != nil {
		t.Fatal(err)
	}
	var other *CUDAContext
	if count > 1 {
		other, err = NewCUDAContextOn(1)
		if err != nil {
			t.Fatal(err)
		}
		if other.Device() != 1 {
			t.Fatalf("second context device %d", other.Device())
		}
		other.Close()
	}
	context.Close()
	if context.Device() != 0 {
		t.Fatalf("device after close %d", context.Device())
	}
	var nilContext *CUDAContext
	nilContext.Close()
	if nilContext.Device() != -1 {
		t.Fatal("nil context reported a device")
	}
}
