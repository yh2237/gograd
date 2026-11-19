//go:build (windows || (linux && !android)) && amd64

package vulkan

import (
	"encoding/binary"
	"errors"
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

const (
	maxBatchDescriptors = 512
	maxBatchOps         = 512
)

func (d *vkDriver) initPipeline() error {
	bindings := [4]vkDescriptorBinding{}
	for i := range bindings {
		bindings[i] = vkDescriptorBinding{Binding: uint32(i), Type: 7, Count: 1, Stage: 0x20}
	}
	layout := vkDescriptorLayoutCreateInfo{vkHeader: header(32), BindingCount: 4, Bindings: &bindings[0]}
	if err := result("vkCreateDescriptorSetLayout", d.call("vkCreateDescriptorSetLayout", d.device, uintptr(unsafe.Pointer(&layout)), 0, uintptr(unsafe.Pointer(&d.descriptorLayout)))); err != nil {
		return err
	}
	d.descriptorSets = make([]uint64, maxBatchDescriptors)
	sizes := vkDescriptorPoolSize{Type: 7, Count: 4 * uint32(len(d.descriptorSets))}
	pool := vkDescriptorPoolCreateInfo{vkHeader: header(33), MaxSets: uint32(len(d.descriptorSets)), SizeCount: 1, Sizes: &sizes}
	if err := result("vkCreateDescriptorPool", d.call("vkCreateDescriptorPool", d.device, uintptr(unsafe.Pointer(&pool)), 0, uintptr(unsafe.Pointer(&d.descriptorPool)))); err != nil {
		return err
	}
	layouts := make([]uint64, len(d.descriptorSets))
	for i := range layouts {
		layouts[i] = d.descriptorLayout
	}
	allocate := vkDescriptorAllocateInfo{vkHeader: header(34), Pool: d.descriptorPool, Count: uint32(len(layouts)), Layouts: &layouts[0]}
	if err := result("vkAllocateDescriptorSets", d.call("vkAllocateDescriptorSets", d.device, uintptr(unsafe.Pointer(&allocate)), uintptr(unsafe.Pointer(&d.descriptorSets[0])))); err != nil {
		return err
	}
	d.descriptorSet = d.descriptorSets[0]
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
	if d.binaryShader, err = d.createShader(binarySPV); err != nil {
		return err
	}
	if d.binaryPipeline, err = d.createCompute(d.binaryShader, nil); err != nil {
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
	if d.ndReduceShader, err = d.createShader(ndreduceSPV); err != nil {
		return err
	}
	if d.ndReducePipeline, err = d.createCompute(d.ndReduceShader, nil); err != nil {
		return err
	}
	if d.visionShader, err = d.createShader(visionSPV); err != nil {
		return err
	}
	if d.visionPipeline, err = d.createCompute(d.visionShader, nil); err != nil {
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

func (d *vkDriver) hasBatch() bool         { return d.recording }
func (d *vkDriver) discard()               { d.recording, d.batchNext, d.batchOps = false, 0, 0 }
func (d *vkDriver) setFlushHook(fn func()) { d.onFlush = fn }

func (d *vkDriver) ensureBatch(slots int) ([]uint64, error) {
	if d.failure != nil {
		return nil, d.failure
	}
	if slots > len(d.descriptorSets) {
		return nil, fmt.Errorf("vulkan: descriptor batch exceeds %d slots", len(d.descriptorSets))
	}
	if d.recording && (d.batchNext+slots > len(d.descriptorSets) || d.batchOps >= maxBatchOps) {
		if err := d.flush(); err != nil {
			return nil, err
		}
	}
	if !d.recording {
		if err := result("vkResetCommandBuffer", d.call("vkResetCommandBuffer", d.command, 0)); err != nil {
			d.failure = err
			return nil, err
		}
		begin := vkCommandBufferBeginInfo{vkHeader: header(42), Flags: 1}
		if err := result("vkBeginCommandBuffer", d.call("vkBeginCommandBuffer", d.command, uintptr(unsafe.Pointer(&begin)))); err != nil {
			d.failure = err
			return nil, err
		}
		d.recording = true
		d.batchNext = 0
		d.batchOps = 0
	}
	sets := d.descriptorSets[d.batchNext : d.batchNext+slots]
	d.batchNext += slots
	return sets, nil
}

// No command state or allocation can be reused until completion is established.
// Failed emergency idle waits leave pending set, quarantining the entire device.
func (d *vkDriver) flush() (err error) {
	if !d.recording {
		return d.failure
	}
	defer func() {
		d.recording = false
		d.batchNext = 0
		d.batchOps = 0
		if err != nil {
			d.failure = err
		}
		if d.onFlush != nil {
			d.onFlush()
		}
	}()
	if err := result("vkEndCommandBuffer", d.call("vkEndCommandBuffer", d.command)); err != nil {
		return err
	}
	if err := result("vkResetFences", d.call("vkResetFences", d.device, 1, uintptr(unsafe.Pointer(&d.fence)))); err != nil {
		return err
	}
	submit := vkSubmitInfo{vkHeader: header(4), CommandCount: 1, Commands: &d.command}
	d.pending = true
	if code := d.call("vkQueueSubmit", d.queue, 1, uintptr(unsafe.Pointer(&submit)), uintptr(d.fence)); code != 0 {
		return d.submissionFailure("vkQueueSubmit", code)
	}
	if code := d.call("vkWaitForFences", d.device, 1, uintptr(unsafe.Pointer(&d.fence)), 1, ^uintptr(0)); code != 0 {
		return d.submissionFailure("vkWaitForFences", code)
	}
	d.pending = false
	return nil
}

func (d *vkDriver) submit(record func()) error {
	if _, err := d.ensureBatch(0); err != nil {
		return err
	}
	record()
	return d.flush()
}

func (d *vkDriver) submissionFailure(op string, code uintptr) error {
	err := result(op, code)
	if int32(code) == -4 { // Confirmed VK_ERROR_DEVICE_LOST: destruction is safe.
		d.pending = false
		return err
	}
	idle := d.call("vkDeviceWaitIdle", d.device)
	if idle == 0 || int32(idle) == -4 {
		d.pending = false
	}
	return errors.Join(err, result("emergency vkDeviceWaitIdle", idle))
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

func (d *vkDriver) beforeCompute(out *block, inputs ...*block) {
	if d.batchOps == 0 {
		for i, in := range inputs {
			seen := false
			for j := 0; j < i; j++ {
				if inputs[j] == in {
					seen = true
					break
				}
			}
			if !seen {
				d.before(in, accessShaderRead, stageCompute)
			}
		}
		d.before(out, accessShaderWrite, stageCompute)
		return
	}
	d.barrier(out, accessShaderRead|accessShaderWrite|accessTransferRead|accessTransferWrite, accessShaderWrite, stageCompute|stageTransfer, stageCompute)
}

func (d *vkDriver) copy(out, src *block, size uint64) error {
	if _, err := d.ensureBatch(0); err != nil {
		return err
	}
	d.before(src, accessTransferRead, stageTransfer)
	d.before(out, accessTransferWrite, stageTransfer)
	region := vkBufferCopy{Size: size}
	d.call("vkCmdCopyBuffer", d.command, uintptr(src.buffer), uintptr(out.buffer), 1, uintptr(unsafe.Pointer(&region)))
	d.afterTransfer(out)
	d.batchOps++
	return nil
}
func (d *vkDriver) zero(out *block, size uint64) error {
	if _, err := d.ensureBatch(0); err != nil {
		return err
	}
	d.before(out, accessTransferWrite, stageTransfer)
	d.call("vkCmdFillBuffer", d.command, uintptr(out.buffer), 0, uintptr(size), 0)
	d.afterTransfer(out)
	d.batchOps++
	return nil
}

func (d *vkDriver) pointwise(out, a, b *block, n, kind uint32, scalar, scalar2 float32) error {
	x, y, err := dispatchGrid(n, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, b, n, n, n)
	d.beforeCompute(out, a, b)
	push := vkPointwisePush{N: n, Kind: kind, Scalar: scalar, Scalar2: scalar2}
	d.bindComputeSet(sets[0], d.pipeline, unsafe.Pointer(&push), unsafe.Sizeof(push))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) bindBuffersSet(set uint64, out, a, b *block, outCount, aCount, bCount uint32) {
	if d.failure != nil {
		return
	}
	// Binding 3 is statically used by pointwise's fused residual branch. Bind a
	// valid dummy even for other operations; a dynamic branch is not sufficient
	// to leave a shader-visible descriptor uninitialized. Reductions override it.
	infos := [4]vkDescriptorBufferInfo{{Buffer: a.buffer, Range: uint64(aCount) * 4}, {Buffer: b.buffer, Range: uint64(bCount) * 4}, {Buffer: out.buffer, Range: uint64(outCount) * 4}, {Buffer: a.buffer, Range: uint64(aCount) * 4}}
	writes := [4]vkDescriptorWrite{}
	for i := range writes {
		writes[i] = vkDescriptorWrite{vkHeader: header(35), Set: set, Binding: uint32(i), Count: 1, Type: 7, BufferInfo: &infos[i]}
	}
	d.call("vkUpdateDescriptorSets", d.device, uintptr(len(writes)), uintptr(unsafe.Pointer(&writes[0])), 0, 0)
}

func (d *vkDriver) fusedBias(out, a, bias, residual *block, n, channels, kind uint32) error {
	x, y, err := dispatchGrid(n, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, bias, n, n, channels)
	info := vkDescriptorBufferInfo{Buffer: residual.buffer, Range: uint64(n) * 4}
	write := vkDescriptorWrite{vkHeader: header(35), Set: sets[0], Binding: 3, Count: 1, Type: 7, BufferInfo: &info}
	d.call("vkUpdateDescriptorSets", d.device, 1, uintptr(unsafe.Pointer(&write)), 0, 0)
	d.beforeCompute(out, a, bias, residual)
	push := vkPointwisePush{N: n, Kind: kind, Channels: channels}
	d.bindComputeSet(sets[0], d.pipeline, unsafe.Pointer(&push), unsafe.Sizeof(push))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) bindComputeSet(set, pipeline uint64, push unsafe.Pointer, size uintptr) {
	d.call("vkCmdBindPipeline", d.command, 1, uintptr(pipeline))
	d.call("vkCmdBindDescriptorSets", d.command, 1, uintptr(d.pipelineLayout), 0, 1, uintptr(unsafe.Pointer(&set)), 0, 0)
	d.call("vkCmdPushConstants", d.command, uintptr(d.pipelineLayout), 0x20, 0, size, uintptr(push))
}

func (d *vkDriver) viewCopy(out, src *block, outCount, srcCount uint32, p viewParams) error {
	x, y, err := dispatchGrid(p.N, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, src, src, outCount, srcCount, srcCount)
	d.beforeCompute(out, src)
	d.bindComputeSet(sets[0], d.viewPipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

// binaryViews evaluates an elementwise operation on two strided views. The
// output is compact; the operands keep their own strides and offsets.
func (d *vkDriver) binaryViews(out, a, b *block, outCount, aCount, bCount uint32, p binaryParams) error {
	x, y, err := dispatchGrid(p.N, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, b, outCount, aCount, bCount)
	d.beforeCompute(out, a, b)
	d.bindComputeSet(sets[0], d.binaryPipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) gemm(out, a, b *block, outCount, aCount, bCount uint32, p gemmParams, variant int) error {
	x, y, err := groupGrid(uint64(p.Batch)*uint64(p.TilesPerMatrix), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, b, outCount, aCount, bCount)
	d.beforeCompute(out, a, b)
	d.bindComputeSet(sets[0], d.gemmPipelines[variant], unsafe.Pointer(&p), unsafe.Sizeof(p))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) reduce(out, a, weight, bias *block, count, affineCount uint32, p reduceParams) error {
	x, y, err := groupGrid(uint64(p.Rows), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, weight, count, count, affineCount)
	info := vkDescriptorBufferInfo{Buffer: bias.buffer, Range: uint64(affineCount) * 4}
	write := vkDescriptorWrite{vkHeader: header(35), Set: sets[0], Binding: 3, Count: 1, Type: 7, BufferInfo: &info}
	d.call("vkUpdateDescriptorSets", d.device, 1, uintptr(unsafe.Pointer(&write)), 0, 0)
	d.beforeCompute(out, a, weight, bias)
	d.bindComputeSet(sets[0], d.reducePipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

// Each dispatch owns an independent descriptor slot. Descriptor writes are not
// command-buffer state: rewriting one set between recorded dispatches is unsafe.
// Push constants, in contrast, are copied into the command stream by Vulkan.
func (d *vkDriver) conv1d(b convBatch) error {
	x, y, err := dispatchGrid(b.colCount, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	gx, gy, err := groupGrid(uint64(b.gemm.Batch)*uint64(b.gemm.TilesPerMatrix), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	slots := 2
	var bx, by uint32
	if b.hasBias {
		slots = 3
		bx, by, err = dispatchGrid(b.outCount, d.properties.MaxWorkGroupCount)
		if err != nil {
			return err
		}
	}
	sets, err := d.ensureBatch(slots)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], b.col, b.input, b.bias, b.colCount, b.inputCount, b.biasCount)
	d.bindBuffersSet(sets[1], b.out, b.col, b.weight, b.outCount, b.colCount, b.weightCount)
	if b.hasBias {
		d.bindBuffersSet(sets[2], b.out, b.out, b.bias, b.outCount, b.outCount, b.biasCount)
	}
	d.beforeCompute(b.col, b.input)
	d.bindComputeSet(sets[0], d.convPipeline, unsafe.Pointer(&b.conv), unsafe.Sizeof(b.conv))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(b.col, accessShaderWrite, accessShaderRead, stageCompute, stageCompute)

	if d.batchOps == 0 {
		d.before(b.weight, accessShaderRead, stageCompute)
		d.before(b.out, accessShaderWrite, stageCompute)
	} else {
		d.barrier(b.out, accessShaderRead|accessShaderWrite|accessTransferRead|accessTransferWrite, accessShaderWrite, stageCompute|stageTransfer, stageCompute)
	}
	d.bindComputeSet(sets[1], d.gemmPipelines[b.variant], unsafe.Pointer(&b.gemm), unsafe.Sizeof(b.gemm))
	d.call("vkCmdDispatch", d.command, uintptr(gx), uintptr(gy), 1)
	if b.hasBias {
		d.barrier(b.out, accessShaderWrite, accessShaderRead|accessShaderWrite, stageCompute, stageCompute)
		if d.batchOps == 0 {
			d.before(b.bias, accessShaderRead, stageCompute)
		}
		p := b.conv
		p.N, p.Kind = b.outCount, 1
		d.bindComputeSet(sets[2], d.convPipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
		d.call("vkCmdDispatch", d.command, uintptr(bx), uintptr(by), 1)
	}
	d.barrier(b.out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) ndReduce(out, a *block, outCount, aCount uint32, p ndReduceParams) error {
	x, y, err := groupGrid(uint64(p.OutN), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, a, outCount, aCount, aCount)
	d.beforeCompute(out, a)
	d.bindComputeSet(sets[0], d.ndReducePipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) vision(out, a, b, c *block, outCount, aCount, bCount, cCount uint32, p visionParams) error {
	x, y, err := dispatchGrid(p.N, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	sets, err := d.ensureBatch(1)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], out, a, b, outCount, aCount, bCount)
	info := vkDescriptorBufferInfo{Buffer: c.buffer, Range: uint64(cCount) * 4}
	write := vkDescriptorWrite{vkHeader: header(35), Set: sets[0], Binding: 3, Count: 1, Type: 7, BufferInfo: &info}
	d.call("vkUpdateDescriptorSets", d.device, 1, uintptr(unsafe.Pointer(&write)), 0, 0)
	d.beforeCompute(out, a, b, c)
	d.bindComputeSet(sets[0], d.visionPipeline, unsafe.Pointer(&p), unsafe.Sizeof(p))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}

func (d *vkDriver) conv2d(b conv2dBatch) error {
	x, y, err := dispatchGrid(b.colCount, d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	gx, gy, err := groupGrid(uint64(b.gemm.Batch)*uint64(b.gemm.TilesPerMatrix), d.properties.MaxWorkGroupCount)
	if err != nil {
		return err
	}
	slots := 2
	var bx, by uint32
	if b.hasBias {
		slots = 3
		bx, by, err = dispatchGrid(b.outCount, d.properties.MaxWorkGroupCount)
		if err != nil {
			return err
		}
	}
	sets, err := d.ensureBatch(slots)
	if err != nil {
		return err
	}
	d.bindBuffersSet(sets[0], b.col, b.input, b.input, b.colCount, b.inputCount, b.inputCount)
	d.bindBuffersSet(sets[1], b.out, b.weight, b.col, b.outCount, b.weightCount, b.colCount)
	if b.hasBias {
		d.bindBuffersSet(sets[2], b.out, b.out, b.bias, b.outCount, b.outCount, b.biasCount)
		info := vkDescriptorBufferInfo{Buffer: b.bias.buffer, Range: uint64(b.biasCount) * 4}
		write := vkDescriptorWrite{vkHeader: header(35), Set: sets[2], Binding: 3, Count: 1, Type: 7, BufferInfo: &info}
		d.call("vkUpdateDescriptorSets", d.device, 1, uintptr(unsafe.Pointer(&write)), 0, 0)
	}
	d.beforeCompute(b.col, b.input)
	d.bindComputeSet(sets[0], d.visionPipeline, unsafe.Pointer(&b.im2col), unsafe.Sizeof(b.im2col))
	d.call("vkCmdDispatch", d.command, uintptr(x), uintptr(y), 1)
	d.barrier(b.col, accessShaderWrite, accessShaderRead, stageCompute, stageCompute)

	if d.batchOps == 0 {
		d.before(b.weight, accessShaderRead, stageCompute)
		d.before(b.out, accessShaderWrite, stageCompute)
	} else {
		d.barrier(b.out, accessShaderRead|accessShaderWrite|accessTransferRead|accessTransferWrite, accessShaderWrite, stageCompute|stageTransfer, stageCompute)
	}
	d.bindComputeSet(sets[1], d.gemmPipelines[b.variant], unsafe.Pointer(&b.gemm), unsafe.Sizeof(b.gemm))
	d.call("vkCmdDispatch", d.command, uintptr(gx), uintptr(gy), 1)
	if b.hasBias {
		d.barrier(b.out, accessShaderWrite, accessShaderRead|accessShaderWrite, stageCompute, stageCompute)
		if d.batchOps == 0 {
			d.before(b.bias, accessShaderRead, stageCompute)
		}
		d.bindComputeSet(sets[2], d.visionPipeline, unsafe.Pointer(&b.biasOp), unsafe.Sizeof(b.biasOp))
		d.call("vkCmdDispatch", d.command, uintptr(bx), uintptr(by), 1)
	}
	d.barrier(b.out, accessShaderWrite, accessShaderRead|accessShaderWrite|accessTransferRead, stageCompute, stageCompute|stageTransfer)
	d.batchOps++
	return nil
}
