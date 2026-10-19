//go:build (windows || (linux && !android)) && amd64

package vulkan

import (
	"encoding/binary"
	"fmt"
	"syscall"
	"unsafe"
)

func (d *vkDriver) initCommands() error {
	pool := vkCommandPoolCreateInfo{vkHeader: header(39), Flags: 2, Family: d.properties.QueueFamily}
	if err := result("vkCreateCommandPool", d.call("vkCreateCommandPool", d.device, uintptr(unsafe.Pointer(&pool)), 0, uintptr(unsafe.Pointer(&d.commandPool)))); err != nil {
		return err
	}
	allocate := vkCommandBufferAllocateInfo{vkHeader: header(40), Pool: d.commandPool, Count: 1}
	if err := result("vkAllocateCommandBuffers", d.call("vkAllocateCommandBuffers", d.device, uintptr(unsafe.Pointer(&allocate)), uintptr(unsafe.Pointer(&d.command)))); err != nil {
		return err
	}
	fence := vkFenceCreateInfo{vkHeader: header(8)}
	return result("vkCreateFence", d.call("vkCreateFence", d.device, uintptr(unsafe.Pointer(&fence)), 0, uintptr(unsafe.Pointer(&d.fence))))
}

func (d *vkDriver) initPipeline() error {
	bindings := [4]vkDescriptorBinding{}
	for i := range bindings {
		bindings[i] = vkDescriptorBinding{Binding: uint32(i), Type: 7, Count: 1, Stage: 0x20}
	}
	layout := vkDescriptorLayoutCreateInfo{vkHeader: header(32), BindingCount: 4, Bindings: &bindings[0]}
	if err := result("vkCreateDescriptorSetLayout", d.call("vkCreateDescriptorSetLayout", d.device, uintptr(unsafe.Pointer(&layout)), 0, uintptr(unsafe.Pointer(&d.descriptorLayout)))); err != nil {
		return err
	}
	sizes := vkDescriptorPoolSize{Type: 7, Count: 4}
	pool := vkDescriptorPoolCreateInfo{vkHeader: header(33), MaxSets: 1, SizeCount: 1, Sizes: &sizes}
	if err := result("vkCreateDescriptorPool", d.call("vkCreateDescriptorPool", d.device, uintptr(unsafe.Pointer(&pool)), 0, uintptr(unsafe.Pointer(&d.descriptorPool)))); err != nil {
		return err
	}
	allocate := vkDescriptorAllocateInfo{vkHeader: header(34), Pool: d.descriptorPool, Count: 1, Layouts: &d.descriptorLayout}
	if err := result("vkAllocateDescriptorSets", d.call("vkAllocateDescriptorSets", d.device, uintptr(unsafe.Pointer(&allocate)), uintptr(unsafe.Pointer(&d.descriptorSet)))); err != nil {
		return err
	}
	push := vkPushRange{Stage: 0x20, Size: 128}
	pipelineLayout := vkPipelineLayoutCreateInfo{vkHeader: header(30), SetLayoutCount: 1, SetLayouts: &d.descriptorLayout, PushCount: 1, PushRanges: &push}
	if err := result("vkCreatePipelineLayout", d.call("vkCreatePipelineLayout", d.device, uintptr(unsafe.Pointer(&pipelineLayout)), 0, uintptr(unsafe.Pointer(&d.pipelineLayout)))); err != nil {
		return err
	}
	var err error
	if d.shader, err = d.createShader(pointwiseSPV); err != nil {
		return err
	}
	if d.pipeline, err = d.createCompute(d.shader, nil); err != nil {
		return err
	}
	if d.viewShader, err = d.createShader(viewSPV); err != nil {
		return err
	}
	if d.viewPipeline, err = d.createCompute(d.viewShader, nil); err != nil {
		return err
	}
	if d.gemmShader, err = d.createShader(gemmSPV); err != nil {
		return err
	}
	entries := [2]vkSpecializationMapEntry{{ID: 0, Size: 4}, {ID: 1, Offset: 4, Size: 4}}
	for variant := range d.gemmPipelines {
		data := [2]uint32{uint32(variant >> 1), uint32(variant & 1)}
		specialization := vkSpecializationInfo{Count: 2, Entries: &entries[0], DataSize: 8, Data: &data[0]}
		if d.gemmPipelines[variant], err = d.createCompute(d.gemmShader, &specialization); err != nil {
			return err
		}
	}
	if d.reduceShader, err = d.createShader(reduceSPV); err != nil {
		return err
	}
	if d.reducePipeline, err = d.createCompute(d.reduceShader, nil); err != nil {
		return err
	}
	if d.convShader, err = d.createShader(convSPV); err != nil {
		return err
	}
	if d.convPipeline, err = d.createCompute(d.convShader, nil); err != nil {
		return err
	}
	return nil
}

func (d *vkDriver) createShader(code []byte) (uint64, error) {
	if len(code)%4 != 0 || len(code) < 20 || binary.LittleEndian.Uint32(code) != 0x07230203 {
		return 0, fmt.Errorf("vulkan: invalid embedded SPIR-V")
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[i*4:])
	}
	create := vkShaderCreateInfo{vkHeader: header(16), CodeSize: uintptr(len(code)), Code: &words[0]}
	var shader uint64
	err := result("vkCreateShaderModule", d.call("vkCreateShaderModule", d.device, uintptr(unsafe.Pointer(&create)), 0, uintptr(unsafe.Pointer(&shader))))
	return shader, err
}

func (d *vkDriver) createCompute(shader uint64, specialization *vkSpecializationInfo) (uint64, error) {
	main, _ := syscall.BytePtrFromString("main")
	stage := vkShaderStage{vkHeader: header(18), Stage: 0x20, Module: shader, Name: main, Specialization: unsafe.Pointer(specialization)}
	pipeline := vkComputePipelineCreateInfo{vkHeader: header(29), Stage: stage, Layout: d.pipelineLayout, BaseIndex: -1}
	var handle uint64
	err := result("vkCreateComputePipelines", d.call("vkCreateComputePipelines", d.device, 0, 1, uintptr(unsafe.Pointer(&pipeline)), 0, uintptr(unsafe.Pointer(&handle))))
	return handle, err
}

// A fence is always waited before command buffers, descriptors or allocations
// can be reused. On submission failure, drain the device before returning an
// error to the pool layer; a lost device permits resource destruction as well.
func (d *vkDriver) submit(record func()) error {
	if err := result("vkResetCommandBuffer", d.call("vkResetCommandBuffer", d.command, 0)); err != nil {
		return err
	}
	begin := vkCommandBufferBeginInfo{vkHeader: header(42), Flags: 1}
	if err := result("vkBeginCommandBuffer", d.call("vkBeginCommandBuffer", d.command, uintptr(unsafe.Pointer(&begin)))); err != nil {
		return err
	}
	record()
	if err := result("vkEndCommandBuffer", d.call("vkEndCommandBuffer", d.command)); err != nil {
		return err
	}
	if err := result("vkResetFences", d.call("vkResetFences", d.device, 1, uintptr(unsafe.Pointer(&d.fence)))); err != nil {
		return err
	}
	submit := vkSubmitInfo{vkHeader: header(4), CommandCount: 1, Commands: &d.command}
	if err := result("vkQueueSubmit", d.call("vkQueueSubmit", d.queue, 1, uintptr(unsafe.Pointer(&submit)), uintptr(d.fence))); err != nil {
		d.call("vkDeviceWaitIdle", d.device)
		return err
	}
	if err := result("vkWaitForFences", d.call("vkWaitForFences", d.device, 1, uintptr(unsafe.Pointer(&d.fence)), 1, ^uintptr(0))); err != nil {
		d.call("vkDeviceWaitIdle", d.device)
		return err
	}
	return nil
}

const (
	stageCompute        = 0x800
	stageTransfer       = 0x1000
	stageHost           = 0x4000
	stageAll            = 0x10000
	accessShaderRead    = 0x20
	accessShaderWrite   = 0x40
	accessTransferRead  = 0x800
	accessTransferWrite = 0x1000
	accessHostRead      = 0x2000
	accessHostWrite     = 0x4000
	accessMemoryRead    = 0x8000
	accessMemoryWrite   = 0x10000
)

func (d *vkDriver) barrier(memory *block, srcAccess, dstAccess, srcStage, dstStage uint32) {
	barrier := vkBufferBarrier{vkHeader: header(44), SrcAccess: srcAccess, DstAccess: dstAccess, SrcFamily: ^uint32(0), DstFamily: ^uint32(0), Buffer: memory.buffer, Size: ^uint64(0)}
	d.call("vkCmdPipelineBarrier", d.command, uintptr(srcStage), uintptr(dstStage), 0, 0, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0)
}

func (d *vkDriver) before(memory *block, access, stage uint32) {
	d.barrier(memory, accessMemoryRead|accessMemoryWrite|accessHostWrite, access, stageAll|stageHost, stage)
}

func (d *vkDriver) afterTransfer(memory *block) {
	if memory.host {
		d.barrier(memory, accessTransferWrite, accessHostRead, stageTransfer, stageHost)
	} else {
		d.barrier(memory, accessTransferWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageTransfer, stageCompute|stageTransfer)
	}
}

func (d *vkDriver) copy(out, src *block, size uint64) error {
	return d.submit(func() {
		d.before(src, accessTransferRead, stageTransfer)
		d.before(out, accessTransferWrite, stageTransfer)
		region := vkBufferCopy{Size: size}
		d.call("vkCmdCopyBuffer", d.command, uintptr(src.buffer), uintptr(out.buffer), 1, uintptr(unsafe.Pointer(&region)))
		d.afterTransfer(out)
	})
}
func (d *vkDriver) zero(out *block, size uint64) error {
	return d.submit(func() {
		d.before(out, accessTransferWrite, stageTransfer)
		d.call("vkCmdFillBuffer", d.command, uintptr(out.buffer), 0, uintptr(size), 0)
		d.afterTransfer(out)
	})
}

func (d *vkDriver) pointwise(out, a, b *block, n, kind uint32, scalar float32) error {
	x, y, err := dispatchGrid(n, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	d.bindBuffers(out, a, b, n, n, n)
	return d.submit(func() {
		d.before(a, accessShaderRead, stageCompute)
		if b != a {
			d.before(b, accessShaderRead, stageCompute)
		}
		d.before(out, accessShaderWrite, stageCompute)
		push := vkPointwisePush{N: n, Kind: kind, Scalar: scalar}
		d.bindCompute(d.pipeline, unsafe.Pointer(&push), unsafe.Sizeof(push))
		d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
		d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	})
}

func (d *vkDriver) bindBuffers(out, a, b *block, outCount, aCount, bCount uint32) {
	// Binding 3 is statically used by pointwise's fused residual branch. Bind a
	// valid dummy even for other operations; a dynamic branch is not sufficient
	// to leave a shader-visible descriptor uninitialized. Reductions override it.
	infos := [4]vkDescriptorBufferInfo{{Buffer: a.buffer, Range: uint64(aCount) * 4}, {Buffer: b.buffer, Range: uint64(bCount) * 4}, {Buffer: out.buffer, Range: uint64(outCount) * 4}, {Buffer: a.buffer, Range: uint64(aCount) * 4}}
	writes := [4]vkDescriptorWrite{}
	for i := range writes {
		writes[i] = vkDescriptorWrite{vkHeader: header(35), Set: d.descriptorSet, Binding: uint32(i), Count: 1, Type: 7, BufferInfo: &infos[i]}
	}
	d.call("vkUpdateDescriptorSets", d.device, uintptr(len(writes)), uintptr(unsafe.Pointer(&writes[0])), 0, 0)
}

func (d *vkDriver) fusedBias(out, a, bias, residual *block, n, channels, kind uint32) error {
	x, y, err := dispatchGrid(n, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	d.bindBuffers(out, a, bias, n, n, channels)
	info := vkDescriptorBufferInfo{Buffer: residual.buffer, Range: uint64(n) * 4}
	write := vkDescriptorWrite{vkHeader: header(35), Set: d.descriptorSet, Binding: 3, Count: 1, Type: 7, BufferInfo: &info}
	d.call("vkUpdateDescriptorSets", d.device, 1, uintptr(unsafe.Pointer(&write)), 0, 0)
	return d.submit(func() {
		d.before(a, accessShaderRead, stageCompute)
		if bias != a {
			d.before(bias, accessShaderRead, stageCompute)
		}
		if residual != a && residual != bias {
			d.before(residual, accessShaderRead, stageCompute)
		}
		d.before(out, accessShaderWrite, stageCompute)
		push := vkPointwisePush{N: n, Kind: kind, Channels: channels}
		d.bindCompute(d.pipeline, unsafe.Pointer(&push), unsafe.Sizeof(push))
		d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
		d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	})
}

func (d *vkDriver) bindCompute(pipeline uint64, push unsafe.Pointer, size uintptr) {
	d.call("vkCmdBindPipeline", d.command, 1, uintptr(pipeline))
	d.call("vkCmdBindDescriptorSets", d.command, 1, uintptr(d.pipelineLayout), 0, 1, uintptr(unsafe.Pointer(&d.descriptorSet)), 0, 0)
	d.call("vkCmdPushConstants", d.command, uintptr(d.pipelineLayout), 0x20, 0, size, uintptr(push))
}

func (d *vkDriver) viewCopy(out, src *block, outCount, srcCount uint32, p viewParams) error {
	x, y, err := dispatchGrid(p.N, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	d.bindBuffers(out, src, src, outCount, srcCount, srcCount)
	return d.submit(func() {
		d.before(src, accessShaderRead, stageCompute)
		d.before(out, accessShaderWrite, stageCompute)
		d.bindCompute(d.viewPipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
		d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
		d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	})
}

func (d *vkDriver) gemm(out, a, b *block, outCount, aCount, bCount uint32, p gemmParams, variant int) error {
	x, y, err := groupGrid(uint64(p.Batch)*uint64(p.TilesPerMatrix), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	d.bindBuffers(out, a, b, outCount, aCount, bCount)
	return d.submit(func() {
		d.before(a, accessShaderRead, stageCompute)
		if b != a {
			d.before(b, accessShaderRead, stageCompute)
		}
		d.before(out, accessShaderWrite, stageCompute)
		d.bindCompute(d.gemmPipelines[variant], unsafe.Pointer(&p), unsafe.Sizeof(p))
		d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
		d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	})
}

func (d *vkDriver) reduce(out, a, weight, bias *block, count, affineCount uint32, p reduceParams) error {
	x, y, err := groupGrid(uint64(p.Rows), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	d.bindBuffers(out, a, weight, count, count, affineCount)
	info := vkDescriptorBufferInfo{Buffer: bias.buffer, Range: uint64(affineCount) * 4}
	write := vkDescriptorWrite{vkHeader: header(35), Set: d.descriptorSet, Binding: 3, Count: 1, Type: 7, BufferInfo: &info}
	d.call("vkUpdateDescriptorSets", d.device, 1, uintptr(unsafe.Pointer(&write)), 0, 0)
	return d.submit(func() {
		d.before(a, accessShaderRead, stageCompute)
		if weight != a {
			d.before(weight, accessShaderRead, stageCompute)
		}
		if bias != a && bias != weight {
			d.before(bias, accessShaderRead, stageCompute)
		}
		d.before(out, accessShaderWrite, stageCompute)
		d.bindCompute(d.reducePipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
		d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
		d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	})
}

func (d *vkDriver) conv(out, src, bias *block, outCount, srcCount, biasCount uint32, p convParams) error {
	x, y, err := dispatchGrid(p.N, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	d.bindBuffers(out, src, bias, outCount, srcCount, biasCount)
	return d.submit(func() {
		d.before(src, accessShaderRead, stageCompute)
		if bias != src {
			d.before(bias, accessShaderRead, stageCompute)
		}
		d.before(out, accessShaderWrite, stageCompute)
		d.bindCompute(d.convPipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
		d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
		d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	})
}
