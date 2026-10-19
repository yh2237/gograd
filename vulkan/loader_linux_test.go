//go:build linux && !android && amd64

package vulkan

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math/bits"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/vulkan/internal/abitest"
)

func TestVulkanLinuxMissingLibrary(t *testing.T) {
	l, err := openLinuxLibrary(filepath.Join(t.TempDir(), "gograd-missing-vulkan.so"))
	if l != nil || !errors.Is(err, gpu.ErrUnavailable) {
		t.Fatalf("library=%v err=%v", l, err)
	}
}

func TestVulkanLinuxLibraryLifetime(t *testing.T) {
	for i := 0; i < 3; i++ {
		l, err := openVulkanLibrary()
		if err != nil {
			t.Skip(err)
		}
		proc, err := l.lookup("vkGetInstanceProcAddr")
		if err != nil {
			l.close()
			t.Fatal(err)
		}
		if proc == 0 {
			l.close()
			t.Fatal("null vkGetInstanceProcAddr")
		}
		if _, err := l.lookup("gograd_vulkan_symbol_that_does_not_exist"); err == nil {
			l.close()
			t.Fatal("missing symbol succeeded")
		}
		l.close()
		l.close()
		if l.handle != 0 {
			t.Fatal("library was not closed")
		}
		if _, err := l.lookup("vkGetInstanceProcAddr"); !errors.Is(err, gpu.ErrClosed) {
			t.Fatal(err)
		}
	}
}

func TestVulkanLinuxDeviceEnumeration(t *testing.T) {
	library, err := openVulkanLibrary()
	if err != nil {
		t.Skip(err)
	}
	d := &vkDriver{library: library, procs: make(map[string]uintptr)}
	defer d.close()
	d.getProc, err = library.lookup("vkGetInstanceProcAddr")
	if err != nil {
		t.Fatal(err)
	}
	if err := d.resolve("vkCreateInstance"); err != nil {
		t.Fatal(err)
	}
	create := vkInstanceCreateInfo{vkHeader: header(1)}
	if err := result("vkCreateInstance", d.call("vkCreateInstance", uintptr(unsafe.Pointer(&create)), 0, uintptr(unsafe.Pointer(&d.instance)))); err != nil {
		t.Skipf("no usable Vulkan ICD: %v", err)
	}
	for _, name := range []string{"vkDestroyInstance", "vkEnumeratePhysicalDevices", "vkGetPhysicalDeviceProperties", "vkGetPhysicalDeviceQueueFamilyProperties"} {
		if err := d.resolve(name); err != nil {
			t.Fatal(err)
		}
	}
	devices, err := d.physicalDevices()
	if err != nil {
		t.Skip(err)
	}
	allCPU := true
	for _, device := range devices {
		var p vkPhysicalProperties
		d.call("vkGetPhysicalDeviceProperties", device, uintptr(unsafe.Pointer(&p)))
		if p.DeviceType != 4 {
			allCPU = false
		}
		end := bytes.IndexByte(p.DeviceName[:], 0)
		if end < 0 {
			end = len(p.DeviceName)
		}
		t.Logf("Vulkan device: %s, type=%d, API=%d.%d.%d", p.DeviceName[:end], p.DeviceType, p.APIVersion>>22, p.APIVersion>>12&1023, p.APIVersion&4095)
	}
	if err := d.selectPhysical(); allCPU && !errors.Is(err, gpu.ErrUnavailable) {
		t.Fatalf("software-only enumeration advertised a GPU: %v", err)
	}
}

// Probe all six integer registers plus stack arguments. Vulkan's barrier has
// ten arguments, and VkDeviceSize/handles require all 64 bits to survive.
func TestVulkanLinuxNativeABI(t *testing.T) {
	d := &vkDriver{procs: map[string]uintptr{"probe": abitest.Address()}}
	probe := func(seed uint64) error {
		var out [2]uint64
		args := [9]uint64{}
		want := uint64(0x1020304050607080)
		for i := range args {
			args[i] = seed + uint64(i)*0x102030405
			want = bits.RotateLeft64(want, 7) ^ args[i]
		}
		got := d.call("probe", uintptr(unsafe.Pointer(&out[0])), uintptr(args[0]), uintptr(args[1]), uintptr(args[2]), uintptr(args[3]), uintptr(args[4]), uintptr(args[5]), uintptr(args[6]), uintptr(args[7]), uintptr(args[8]))
		if uint64(got) != want || out[0] != want || out[1] != 8 {
			return fmt.Errorf("native ABI: return=%x memory=%x alignment=%d want=%x", got, out[0], out[1], want)
		}
		return nil
	}
	if err := probe(0xfedcba9876543210); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				if err := probe(uint64(worker)<<60 | uint64(i)); err != nil {
					t.Error(err)
					return
				}
			}
		}(worker)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			runtime.GC()
		}
	}()
	wg.Wait()
}

func TestVulkanLinuxLoaderSubprocess(t *testing.T) {
	switch os.Getenv("GOGRAD_VULKAN_LOADER_TEST") {
	case "lazy":
		// A fresh process imports and registers the backend but never calls
		// Available. Neither Vulkan nor its ICD may be loaded just by import.
		if _, ok := gpu.Lookup("vulkan"); !ok {
			t.Fatal("backend not registered")
		}
		if s := defaultBackend.MemoryStats(); s.ReservedBytes != 0 {
			t.Fatal(s)
		}
		data, err := os.ReadFile("/proc/self/maps")
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), "libvulkan.so") {
			t.Fatal("import eagerly loaded Vulkan")
		}
		return
	case "no-icd":
		for i := 0; i < 3; i++ {
			b := NewBackend()
			if b.Available() || !errors.Is(b.Err(), gpu.ErrUnavailable) {
				t.Fatalf("missing ICD did not report unavailable: %v", b.Err())
			}
			if _, err := b.Alloc(1); !errors.Is(err, gpu.ErrUnavailable) {
				t.Fatal(err)
			}
			if stats := b.MemoryStats(); stats.ReservedBytes != 0 || stats.LiveBuffers != 0 {
				t.Fatal(stats)
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
		}
		return
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"lazy", "no-icd"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, executable, "-test.run=^TestVulkanLinuxLoaderSubprocess$", "-test.v")
			for _, e := range os.Environ() {
				if !strings.HasPrefix(e, "GOGRAD_VULKAN_LOADER_TEST=") && !strings.HasPrefix(e, "VK_DRIVER_FILES=") && !strings.HasPrefix(e, "VK_ICD_FILENAMES=") {
					cmd.Env = append(cmd.Env, e)
				}
			}
			cmd.Env = append(cmd.Env, "GOGRAD_VULKAN_LOADER_TEST="+mode)
			if mode == "no-icd" {
				missing := filepath.Join(t.TempDir(), "missing-icd.json")
				cmd.Env = append(cmd.Env, "VK_DRIVER_FILES="+missing, "VK_ICD_FILENAMES="+missing)
			}
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("%v\n%s", err, output)
			}
		})
	}
}
