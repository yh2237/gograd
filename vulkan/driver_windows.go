//go:build windows && amd64

package vulkan

import (
	"bytes"
	"fmt"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

type vkDriver struct {
	dll                                             *syscall.DLL
	getProc                                         *syscall.Proc
	procs                                           map[string]uintptr
	instance, physical, device, queue, command      uintptr
	commandPool, fence                              uint64
	descriptorLayout, descriptorPool, descriptorSet uint64
	pipelineLayout, shader, pipeline                uint64
	viewShader, viewPipeline                        uint64
	gemmShader                                      uint64
	gemmPipelines                                   [4]uint64
	properties                                      Info
	memory                                          vkMemoryProperties
}

func result(op string, value uintptr) error {
	if code := int32(value); code != 0 {
		return fmt.Errorf("vulkan: %s returned VkResult %d", op, code)
	}
	return nil
}

//go:uintptrescapes
func (d *vkDriver) call(name string, args ...uintptr) uintptr {
	r, _, _ := syscall.SyscallN(d.procs[name], args...)
	return r
}

func (d *vkDriver) resolve(name string) error {
	cname, _ := syscall.BytePtrFromString(name)
	proc, _, _ := d.getProc.Call(d.instance, uintptr(unsafe.Pointer(cname)))
	runtime.KeepAlive(cname)
	if proc == 0 {
		return fmt.Errorf("%w: missing Vulkan function %s", gpu.ErrUnavailable, name)
	}
	d.procs[name] = proc
	return nil
}

func openDriver() (_ driver, err error) {
	if err := checkABI(); err != nil {
		return nil, err
	}
	// Load from the system directory rather than the application's working
	// directory. kernel32 is a Windows KnownDLL.
	var system [32768]uint16
	getSystem := syscall.NewLazyDLL("kernel32.dll").NewProc("GetSystemDirectoryW")
	n, _, e := getSystem.Call(uintptr(unsafe.Pointer(&system[0])), uintptr(len(system)))
	if n == 0 || n >= uintptr(len(system)) {
		return nil, fmt.Errorf("%w: GetSystemDirectoryW: %v", gpu.ErrUnavailable, e)
	}
	path := filepath.Join(syscall.UTF16ToString(system[:n]), "vulkan-1.dll")
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", gpu.ErrUnavailable, err)
	}
	d := &vkDriver{dll: dll, procs: make(map[string]uintptr)}
	defer func() {
		if err != nil {
			d.close()
		}
	}()
	d.getProc, err = dll.FindProc("vkGetInstanceProcAddr")
	if err != nil {
		return nil, fmt.Errorf("%w: %v", gpu.ErrUnavailable, err)
	}
	if err = d.resolve("vkCreateInstance"); err != nil {
		return nil, err
	}
	name, _ := syscall.BytePtrFromString("gograd")
	app := vkApplicationInfo{vkHeader: header(0), ApplicationName: name, APIVersion: 1<<22 | 1<<12}
	create := vkInstanceCreateInfo{vkHeader: header(1), ApplicationInfo: &app}
	if err = result("vkCreateInstance", d.call("vkCreateInstance", uintptr(unsafe.Pointer(&create)), 0, uintptr(unsafe.Pointer(&d.instance)))); err != nil {
		return nil, fmt.Errorf("%w: %v", gpu.ErrUnavailable, err)
	}
	for _, function := range []string{
		"vkDestroyInstance", "vkEnumeratePhysicalDevices", "vkGetPhysicalDeviceProperties", "vkGetPhysicalDeviceQueueFamilyProperties", "vkGetPhysicalDeviceMemoryProperties",
		"vkCreateDevice", "vkDestroyDevice", "vkGetDeviceQueue", "vkDeviceWaitIdle", "vkQueueSubmit",
		"vkCreateBuffer", "vkDestroyBuffer", "vkGetBufferMemoryRequirements", "vkAllocateMemory", "vkFreeMemory", "vkBindBufferMemory", "vkMapMemory", "vkUnmapMemory",
		"vkCreateCommandPool", "vkDestroyCommandPool", "vkAllocateCommandBuffers", "vkResetCommandBuffer", "vkBeginCommandBuffer", "vkEndCommandBuffer",
		"vkCreateFence", "vkDestroyFence", "vkResetFences", "vkWaitForFences", "vkCmdPipelineBarrier", "vkCmdCopyBuffer", "vkCmdFillBuffer",
		"vkCreateDescriptorSetLayout", "vkDestroyDescriptorSetLayout", "vkCreateDescriptorPool", "vkDestroyDescriptorPool", "vkAllocateDescriptorSets", "vkUpdateDescriptorSets",
		"vkCreatePipelineLayout", "vkDestroyPipelineLayout", "vkCreateShaderModule", "vkDestroyShaderModule", "vkCreateComputePipelines", "vkDestroyPipeline",
		"vkCmdBindPipeline", "vkCmdBindDescriptorSets", "vkCmdPushConstants", "vkCmdDispatch",
	} {
		if err = d.resolve(function); err != nil {
			return nil, err
		}
	}
	if err = d.selectPhysical(); err != nil {
		return nil, err
	}
	priority := float32(1)
	q := vkQueueCreateInfo{vkHeader: header(2), Family: d.properties.QueueFamily, Count: 1, Priorities: &priority}
	dev := vkDeviceCreateInfo{vkHeader: header(3), QueueCount: 1, Queues: &q}
	if err = result("vkCreateDevice", d.call("vkCreateDevice", d.physical, uintptr(unsafe.Pointer(&dev)), 0, uintptr(unsafe.Pointer(&d.device)))); err != nil {
		return nil, err
	}
	d.call("vkGetDeviceQueue", d.device, uintptr(d.properties.QueueFamily), 0, uintptr(unsafe.Pointer(&d.queue)))
	if d.queue == 0 {
		return nil, fmt.Errorf("%w: Vulkan compute queue is null", gpu.ErrUnavailable)
	}
	d.call("vkGetPhysicalDeviceMemoryProperties", d.physical, uintptr(unsafe.Pointer(&d.memory)))
	if d.memory.TypeCount > 32 || d.memory.HeapCount > 16 {
		return nil, fmt.Errorf("%w: invalid Vulkan memory properties", gpu.ErrUnavailable)
	}
	if err = d.initCommands(); err != nil {
		return nil, err
	}
	if err = d.initPipeline(); err != nil {
		return nil, err
	}
	return d, nil
}

func (d *vkDriver) physicalDevices() ([]uintptr, error) {
	for attempt := 0; attempt < 8; attempt++ {
		var count uint32
		if err := result("vkEnumeratePhysicalDevices", d.call("vkEnumeratePhysicalDevices", d.instance, uintptr(unsafe.Pointer(&count)), 0)); err != nil {
			return nil, err
		}
		if count == 0 {
			return nil, fmt.Errorf("%w: no Vulkan physical devices", gpu.ErrUnavailable)
		}
		if count > 4096 {
			return nil, fmt.Errorf("%w: invalid Vulkan device count", gpu.ErrUnavailable)
		}
		devices := make([]uintptr, count)
		code := d.call("vkEnumeratePhysicalDevices", d.instance, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&devices[0])))
		if int32(code) == 5 {
			continue
		} // VK_INCOMPLETE: count changed; retry.
		if err := result("vkEnumeratePhysicalDevices", code); err != nil {
			return nil, err
		}
		return devices[:count], nil
	}
	return nil, fmt.Errorf("%w: unstable Vulkan physical device enumeration", gpu.ErrUnavailable)
}

func (d *vkDriver) selectPhysical() error {
	devices, err := d.physicalDevices()
	if err != nil {
		return err
	}
	best := -1
	for _, physical := range devices {
		var p vkPhysicalProperties
		d.call("vkGetPhysicalDeviceProperties", physical, uintptr(unsafe.Pointer(&p)))
		// Never silently advertise a software/CPU Vulkan device as a GPU.
		if p.DeviceType == 4 || p.APIVersion < 1<<22|1<<12 || p.Limits.MaxComputeWorkGroupInvocations < 128 || p.Limits.MaxComputeWorkGroupSize[0] < 128 || p.Limits.MaxComputeWorkGroupSize[1] < 8 || p.Limits.MaxComputeSharedMemorySize < 1536 || p.Limits.MaxPushConstantsSize < 128 {
			continue
		}
		var count uint32
		d.call("vkGetPhysicalDeviceQueueFamilyProperties", physical, uintptr(unsafe.Pointer(&count)), 0)
		if count == 0 || count > 4096 {
			continue
		}
		families := make([]vkQueueFamily, count)
		d.call("vkGetPhysicalDeviceQueueFamilyProperties", physical, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&families[0])))
		for i, family := range families[:count] {
			if family.Count == 0 || family.Flags&2 == 0 {
				continue
			}
			score := 10
			if p.DeviceType == 1 {
				score = 50
			}
			if p.DeviceType == 2 {
				score = 100
			}
			if family.Flags&1 == 0 {
				score++
			} // Prefer a compute-only family.
			if score <= best {
				continue
			}
			end := bytes.IndexByte(p.DeviceName[:], 0)
			if end < 0 {
				end = len(p.DeviceName)
			}
			best, d.physical = score, physical
			d.properties = Info{Name: string(p.DeviceName[:end]), APIVersion: p.APIVersion, VendorID: p.VendorID, DeviceID: p.DeviceID, QueueFamily: uint32(i), MaxStorageBufferBytes: uint64(p.Limits.MaxStorageBufferRange), MaxWorkGroupCount: p.Limits.MaxComputeWorkGroupCount}
		}
	}
	if best < 0 {
		return fmt.Errorf("%w: no Vulkan 1.1 compute GPU", gpu.ErrUnavailable)
	}
	return nil
}

func (d *vkDriver) info() Info { return d.properties }

func (d *vkDriver) close() {
	if d.device != 0 {
		d.call("vkDeviceWaitIdle", d.device)
		for _, pipeline := range d.gemmPipelines {
			if pipeline != 0 {
				d.call("vkDestroyPipeline", d.device, uintptr(pipeline), 0)
			}
		}
		for _, item := range []struct {
			name   string
			handle uint64
		}{
			{"vkDestroyPipeline", d.pipeline}, {"vkDestroyPipeline", d.viewPipeline},
			{"vkDestroyShaderModule", d.shader}, {"vkDestroyShaderModule", d.viewShader}, {"vkDestroyShaderModule", d.gemmShader}, {"vkDestroyPipelineLayout", d.pipelineLayout},
			{"vkDestroyDescriptorPool", d.descriptorPool}, {"vkDestroyDescriptorSetLayout", d.descriptorLayout},
			{"vkDestroyFence", d.fence}, {"vkDestroyCommandPool", d.commandPool},
		} {
			if item.handle != 0 {
				d.call(item.name, d.device, uintptr(item.handle), 0)
			}
		}
		d.call("vkDestroyDevice", d.device, 0)
		d.device = 0
	}
	if d.instance != 0 && d.procs["vkDestroyInstance"] != 0 {
		d.call("vkDestroyInstance", d.instance, 0)
		d.instance = 0
	}
	if d.dll != nil {
		d.dll.Release()
		d.dll = nil
	}
}
