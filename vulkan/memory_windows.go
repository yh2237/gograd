//go:build windows && amd64

package vulkan

import (
	"fmt"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

func (d *vkDriver) alloc(size uint64, host bool) (_ *block, err error) {
	memory := &block{size: size, host: host}
	defer func() {
		if err != nil {
			d.free(memory)
		}
	}()
	create := vkBufferCreateInfo{vkHeader: header(12), Size: size, Usage: 1 | 2 | 0x20}
	if err = result("vkCreateBuffer", d.call("vkCreateBuffer", d.device, uintptr(unsafe.Pointer(&create)), 0, uintptr(unsafe.Pointer(&memory.buffer)))); err != nil {
		return nil, err
	}
	var req vkMemoryRequirements
	d.call("vkGetBufferMemoryRequirements", d.device, uintptr(memory.buffer), uintptr(unsafe.Pointer(&req)))
	flags := uint32(1) // DEVICE_LOCAL
	if host {
		flags = 2 | 4
	} // HOST_VISIBLE | HOST_COHERENT
	index := -1
	for i := uint32(0); i < d.memory.TypeCount; i++ {
		if req.TypeBits&(1<<i) != 0 && d.memory.Types[i].Flags&flags == flags {
			index = int(i)
			break
		}
	}
	if index < 0 {
		return nil, fmt.Errorf("%w: Vulkan memory type %#x", gpu.ErrUnavailable, flags)
	}
	allocate := vkMemoryAllocateInfo{vkHeader: header(5), Size: req.Size, TypeIndex: uint32(index)}
	if err = result("vkAllocateMemory", d.call("vkAllocateMemory", d.device, uintptr(unsafe.Pointer(&allocate)), 0, uintptr(unsafe.Pointer(&memory.memory)))); err != nil {
		return nil, err
	}
	memory.allocated = req.Size
	if err = result("vkBindBufferMemory", d.call("vkBindBufferMemory", d.device, uintptr(memory.buffer), uintptr(memory.memory), 0)); err != nil {
		return nil, err
	}
	if host {
		if err = result("vkMapMemory", d.call("vkMapMemory", d.device, uintptr(memory.memory), 0, uintptr(req.Size), 0, uintptr(unsafe.Pointer(&memory.mapped)))); err != nil {
			return nil, err
		}
	}
	return memory, nil
}

func (d *vkDriver) free(memory *block) {
	if memory.mapped != nil {
		d.call("vkUnmapMemory", d.device, uintptr(memory.memory))
		memory.mapped = nil
	}
	if memory.buffer != 0 {
		d.call("vkDestroyBuffer", d.device, uintptr(memory.buffer), 0)
		memory.buffer = 0
	}
	if memory.memory != 0 {
		d.call("vkFreeMemory", d.device, uintptr(memory.memory), 0)
		memory.memory = 0
	}
}

func (d *vkDriver) write(memory *block, values []float32) {
	copy(unsafe.Slice((*float32)(memory.mapped), len(values)), values)
}
func (d *vkDriver) read(memory *block, values []float32) {
	copy(values, unsafe.Slice((*float32)(memory.mapped), len(values)))
}
