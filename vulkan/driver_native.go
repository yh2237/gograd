//go:build (windows || (linux && !android)) && amd64

package vulkan

import (
	"bytes"
	"fmt"
	"runtime"
	"syscall"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

type vkDriver struct {
	library                                                *vkLibrary
	getProc                                                uintptr
	procs                                                  map[string]uintptr
	instance, physical, device, queue, command             uintptr
	commandPool, fence                                     uint64
	descriptorLayout, descriptorPool, descriptorSet        uint64
	descriptorSets                                         [3]uint64
	pipelineLayout, shader, pipeline                       uint64
	viewShader, viewPipeline                               uint64
	gemmShader                                             uint64
	gemmPipelines                                          [4]uint64
	reduceShader, reducePipeline, convShader, convPipeline uint64
	properties                                             Info
	memory                                                 vkMemoryProperties
	// pending means completion has not been established, even after an emergency
	// wait. Never reset commands, update descriptors, free memory or unload while set.
	pending    bool
	failure    error
	quarantine []*block
	callHook   func(string, ...uintptr) uintptr // driver-free native tests
}

func result(op string, value uintptr) error {
	if code := int32(value); code != 0 {
		return fmt.Errorf("vulkan: %s returned VkResult %d", op, code)
	}
	return nil
}

//go:uintptrescapes
func (d *vkDriver) call(name string, args ...uintptr) uintptr {
	if d.callHook != nil {
		return d.callHook(name, args...)
	}
	return callNative(d.procs[name], args...)
}

func (d *vkDriver) resolve(name string) error {
	cname, _ := syscall.BytePtrFromString(name)
	proc := callNative(d.getProc, d.instance, uintptr(unsafe.Pointer(cname)))
	runtime.KeepAlive(cname)
	if proc == 0 {
		return fmt.Errorf("%w: missing Vulkan function %s", gpu.ErrUnavailable, name)
	}
	d.procs[name] = proc
	return nil
}

// deviceScore ranks the candidate devices. A discrete GPU outranks an
// integrated one, and a compute-only queue family outranks a combined one.
// The device types are the Vulkan ones: 1 is an integrated GPU, 2 a discrete
// GPU and 4 a CPU, which is rejected before the scoring.
func deviceScore(deviceType uint32, queueFlags uint32) int {
	score := 10
	switch deviceType {
	case 1:
		score = 50
	case 2:
		score = 100
	}
	if queueFlags&1 == 0 {
		score++
	}
	return score
}

func openDriver() (_ driver, err error) {
	if err := checkABI(); err != nil {
		return nil, err
	}
	library, err := openVulkanLibrary()
	if err != nil {
		return nil, err
	}
	d := &vkDriver{library: library, procs: make(map[string]uintptr)}
	defer func() {
		if err != nil {
			d.close()
		}
	}()
	d.getProc, err = library.lookup("vkGetInstanceProcAddr")
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
			score := deviceScore(p.DeviceType, family.Flags)
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

func (d *vkDriver) close() error {
	if d.device != 0 {
		if d.pending {
			code := d.call("vkDeviceWaitIdle", d.device)
			if code != 0 && int32(code) != -4 { // VK_ERROR_DEVICE_LOST also permits destruction.
				return fmt.Errorf("%w: cleanup quarantined: %w", gpu.ErrUnavailable, result("vkDeviceWaitIdle", code))
			}
			d.pending = false
		}
		for _, memory := range d.quarantine {
			d.free(memory)
		}
		d.quarantine = nil
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
			{"vkDestroyPipeline", d.reducePipeline}, {"vkDestroyPipeline", d.convPipeline},
			{"vkDestroyShaderModule", d.reduceShader}, {"vkDestroyShaderModule", d.convShader},
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
	if d.library != nil {
		d.library.close()
		d.library = nil
	}
	return nil
}
