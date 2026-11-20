//go:build windows && amd64

package directml

import (
	"math"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

type stepKind uint8

const (
	stepDispatch stepKind = iota
	stepUnaryShader
	stepCopy
	stepCopyView
)

type queuedStep struct {
	kind    stepKind
	key     opKey
	out     *allocation
	src     *allocation
	size    uint64
	inputs  []resourceBinding
	view    gpu.View
	run     int
	regions int
	scatter bool
}

type compiled struct {
	operator              *object
	temporary, persistent *allocation
	execDescriptors       uint32
}

func (d *nativeDriver) closeOperator(op *compiled) {
	if op.temporary != nil {
		d.free(op.temporary)
	}
	if op.persistent != nil {
		d.free(op.persistent)
	}
	release(op.operator)
}
func (d *nativeDriver) operatorStats() OperatorStats {
	var s OperatorStats
	s.CachedOperators = len(d.operators)
	for _, op := range d.operators {
		if op.temporary != nil {
			s.TemporaryBytes += op.temporary.allocated
		}
		if op.persistent != nil {
			s.PersistentBytes += op.persistent.allocated
		}
	}
	return s
}
func (d *nativeDriver) releaseOperators() {
	for i, pipeline := range d.unaryPipelines {
		release(pipeline)
		d.unaryPipelines[i] = nil
	}
	release(d.unaryRoot)
	d.unaryRoot = nil
	for _, op := range d.operators {
		d.closeOperator(op)
	}
	d.operators = nil
	d.order = nil
}

// Tensor descriptors only need to survive CreateOperator. Each descriptor uses
// four axes to support the baseline DirectML feature level, without copying data.
type tensorSpec struct {
	sizes   [4]uint32
	strides [4]uint32
	buffer  bufferTensorDesc
	tensor  tensorDesc
}

func newTensor(rows, cols int) *tensorSpec {
	t := &tensorSpec{sizes: [4]uint32{1, 1, uint32(rows), uint32(cols)}}
	t.buffer = bufferTensorDesc{DataType: 1, Dimensions: 4, Sizes: &t.sizes[0], Bytes: uint64(rows) * uint64(cols) * 4}
	t.tensor = tensorDesc{Type: 1, Desc: &t.buffer}
	return t
}

func layoutTensor(layout tensorLayout) *tensorSpec {
	t := &tensorSpec{sizes: layout.sizes, strides: layout.strides}
	span := uint64(1)
	for i, dim := range t.sizes {
		span += uint64(dim-1) * uint64(t.strides[i])
	}
	t.buffer = bufferTensorDesc{DataType: 1, Dimensions: 4, Sizes: &t.sizes[0], Strides: &t.strides[0], Bytes: span * 4}
	t.tensor = tensorDesc{Type: 1, Desc: &t.buffer}
	return t
}

func (d *nativeDriver) compile(key opKey) (_ *compiled, err error) {
	op := &compiled{}
	defer func() {
		if err != nil {
			d.closeOperator(op)
		}
	}()
	a, b, out := newTensor(1, key.n), newTensor(1, key.n), newTensor(1, key.n)
	if key.aLayout.sizes[0] != 0 {
		a = layoutTensor(key.aLayout)
	}
	if key.bLayout.sizes[0] != 0 {
		b = layoutTensor(key.bLayout)
	}
	if key.yLayout.sizes[0] != 0 {
		out = layoutTensor(key.yLayout)
	}
	var desc operatorDesc
	scale := scaleBias{Scale: math.Float32frombits(key.scalar), Bias: math.Float32frombits(key.bias)}
	binary := binaryDesc{A: &a.tensor, B: &b.tensor, Out: &out.tensor}
	identity := identityDesc{In: &a.tensor, Out: &out.tensor, ScaleBias: &scale}
	gemm := gemmDesc{Alpha: 1}
	unary := unaryDesc{In: &a.tensor, Out: &out.tensor}
	norm := normDesc{In: &a.tensor, Out: &out.tensor, NormalizeVariance: 1, Epsilon: math.Float32frombits(key.scalar)}
	var convolution convDesc
	var convBias *tensorSpec
	strides, dilations, start, end, outputPadding := [2]uint32{1, uint32(key.conv.Stride)}, [2]uint32{1, uint32(key.conv.Dilation)}, [2]uint32{0, uint32(key.conv.Padding)}, [2]uint32{0, uint32(key.conv.Padding)}, [2]uint32{}
	switch key.kind {
	case opAdd:
		desc = operatorDesc{Type: 4, Desc: unsafe.Pointer(&binary)}
	case opMul:
		desc = operatorDesc{Type: 24, Desc: unsafe.Pointer(&binary)}
	case opScalar:
		desc = operatorDesc{Type: 1, Desc: unsafe.Pointer(&identity)}
	case opIdentity:
		identity.ScaleBias = nil // A view copy must not change signed zero or NaN bits.
		desc = operatorDesc{Type: 1, Desc: unsafe.Pointer(&identity)}
	case opGEMM:
		if key.aLayout.sizes[0] == 0 {
			a, b, out = newTensor(key.m, key.k), newTensor(key.k, key.n), newTensor(key.m, key.n)
			if key.ta {
				a = newTensor(key.k, key.m)
				gemm.TransA = 1
			}
			if key.tb {
				b = newTensor(key.n, key.k)
				gemm.TransB = 1
			}
			batch := uint32(max(1, key.batch))
			a.sizes[0], b.sizes[0], out.sizes[0] = batch, batch, batch
			a.buffer.Bytes *= uint64(batch)
			b.buffer.Bytes *= uint64(batch)
			out.buffer.Bytes *= uint64(batch)
		}
		gemm.A, gemm.B, gemm.Out = &a.tensor, &b.tensor, &out.tensor
		desc = operatorDesc{Type: 54, Desc: unsafe.Pointer(&gemm)}
	case opTanh:
		desc = operatorDesc{Type: 51, Desc: unsafe.Pointer(&unary)}
	case opGELUExact:
		desc = operatorDesc{Type: 157, Desc: unsafe.Pointer(&unary)}
	case opSoftmax:
		desc = operatorDesc{Type: 48, Desc: unsafe.Pointer(&unary)}
	case opNorm:
		desc = operatorDesc{Type: 73, Desc: unsafe.Pointer(&norm)}
	case opDiv, opMax, opMin, opSub:
		desc = operatorDesc{Type: key.kind - sdkOp, Desc: unsafe.Pointer(&binary)}
	case opAbs, opCos, opExp, opLog, opSin, opSqrt, opErf:
		identity.ScaleBias = nil
		desc = operatorDesc{Type: key.kind - sdkOp, Desc: unsafe.Pointer(&identity)}
	case opReLU, opSigmoid:
		desc = operatorDesc{Type: key.kind - sdkOp, Desc: unsafe.Pointer(&unary)}
	case opSoftplus:
		softplus := softplusDesc{In: &a.tensor, Out: &out.tensor, Steepness: 1}
		desc = operatorDesc{Type: 49, Desc: unsafe.Pointer(&softplus)}
	case opReduce:
		var axes [4]uint32
		var count uint32
		for axis := range axes {
			if key.bias&(1<<axis) != 0 {
				axes[count] = uint32(axis)
				count++
			}
		}
		reduce := reduceDesc{Function: key.scalar, In: &a.tensor, Out: &out.tensor, AxisCount: count, Axes: &axes[0]}
		desc = operatorDesc{Type: 55, Desc: unsafe.Pointer(&reduce)}
	case opConv2d:
		s := &key.spatial
		convolution = convDesc{In: &a.tensor, Filter: &b.tensor, Out: &out.tensor, Mode: 1, Dimensions: 2,
			Strides: &s.stride[0], Dilations: &s.dilation[0], Start: &s.start[0], End: &s.end[0],
			OutputPadding: &outputPadding[0], Groups: key.scalar}
		if key.hasBias {
			convBias = layoutTensor(compactLayout(1, key.yLayout.sizes[1], 1, 1))
			convolution.Bias = &convBias.tensor
		}
		desc = operatorDesc{Type: 53, Desc: unsafe.Pointer(&convolution)}
	case opAvgPool:
		s := &key.spatial
		pool := avgPoolDesc{In: &a.tensor, Out: &out.tensor, Dimensions: 2, Strides: &s.stride[0],
			Window: &s.window[0], StartPad: &s.start[0], EndPad: &s.end[0], IncludePad: key.scalar}
		desc = operatorDesc{Type: 56, Desc: unsafe.Pointer(&pool)}
	case opMaxPool:
		s := &key.spatial
		pool := maxPoolDesc{In: &a.tensor, Out: &out.tensor, Dimensions: 2, Strides: &s.stride[0],
			Window: &s.window[0], StartPad: &s.start[0], EndPad: &s.end[0]}
		desc = operatorDesc{Type: 58, Desc: unsafe.Pointer(&pool)}
	case opMaxPool2:
		s := &key.spatial
		pool := maxPool2Desc{In: &a.tensor, Out: &out.tensor, Dimensions: 2, Strides: &s.stride[0],
			Window: &s.window[0], StartPad: &s.start[0], EndPad: &s.end[0], Dilations: &s.dilation[0]}
		desc = operatorDesc{Type: 110, Desc: unsafe.Pointer(&pool)}
	case opBatchNorm:
		bn := batchNormDesc{In: &a.tensor, Mean: &b.tensor, Variance: &b.tensor, Scale: &b.tensor,
			Bias: &b.tensor, Out: &out.tensor, Spatial: 1, Epsilon: scale.Scale}
		desc = operatorDesc{Type: 72, Desc: unsafe.Pointer(&bn)}
	case opConv:
		o := key.conv
		a = layoutTensor(ntcLayout(o.Batch, o.InputLength, o.InChannels))
		b = layoutTensor(compactLayout(uint32(o.OutChannels), uint32(o.InChannels), 1, uint32(o.Kernel)))
		out = layoutTensor(ntcLayout(o.Batch, key.n, o.OutChannels))
		convolution = convDesc{In: &a.tensor, Filter: &b.tensor, Out: &out.tensor, Mode: 1, Direction: 0, Dimensions: 2, Strides: &strides[0], Dilations: &dilations[0], Start: &start[0], End: &end[0], OutputPadding: &outputPadding[0], Groups: 1}
		if key.hasBias {
			convBias = layoutTensor(compactLayout(1, uint32(o.OutChannels), 1, 1))
			convolution.Bias = &convBias.tensor
		}
		desc = operatorDesc{Type: 53, Desc: unsafe.Pointer(&convolution)}
	}
	var abstract *object
	if err := hresult("CreateOperator", call(d.dml, 8, uintptr(unsafe.Pointer(&desc)), uintptr(unsafe.Pointer(iidOperator)), uintptr(unsafe.Pointer(&abstract)))); err != nil {
		return nil, err
	}
	defer release(abstract)
	// No half-precision flags: all operators request float32 semantics.
	if err := hresult("CompileOperator", call(d.dml, 9, uintptr(unsafe.Pointer(abstract)), 0, uintptr(unsafe.Pointer(iidCompiled)), uintptr(unsafe.Pointer(&op.operator)))); err != nil {
		return nil, err
	}
	var initializer *object
	if err := hresult("CreateOperatorInitializer", call(d.dml, 10, 1, uintptr(unsafe.Pointer(&op.operator)), uintptr(unsafe.Pointer(iidInitializer)), uintptr(unsafe.Pointer(&initializer)))); err != nil {
		return nil, err
	}
	defer release(initializer)
	var initProps, execProps bindingProperties
	call(initializer, 8, uintptr(unsafe.Pointer(&initProps)))
	call(op.operator, 8, uintptr(unsafe.Pointer(&execProps)))
	initDescs := max(1, initProps.Descriptors)
	op.execDescriptors = max(1, execProps.Descriptors)
	if err := d.ensureBatchHeap(max(initDescs, op.execDescriptors)); err != nil {
		return nil, err
	}
	var initTemp *allocation
	if initProps.Temporary != 0 {
		if execProps.Temporary >= initProps.Temporary {
			op.temporary, err = d.alloc(execProps.Temporary, heapDefault)
			if err != nil {
				return nil, err
			}
			initTemp = op.temporary
		} else {
			initTemp, err = d.alloc(initProps.Temporary, heapDefault)
			if err != nil {
				return nil, err
			}
			defer d.free(initTemp)
			if execProps.Temporary != 0 {
				op.temporary, err = d.alloc(execProps.Temporary, heapDefault)
				if err != nil {
					return nil, err
				}
			}
		}
	} else if execProps.Temporary != 0 {
		op.temporary, err = d.alloc(execProps.Temporary, heapDefault)
		if err != nil {
			return nil, err
		}
	}
	if execProps.Persistent != 0 {
		op.persistent, err = d.alloc(execProps.Persistent, heapDefault)
		if err != nil {
			return nil, err
		}
	}
	initDesc := bindingTableDesc{Dispatchable: initializer, CPU: d.batchCPU, GPU: d.batchGPU, Count: initDescs}
	if err := hresult("Reset(init table)", call(d.batchTable, 12, uintptr(unsafe.Pointer(&initDesc)))); err != nil {
		return nil, err
	}
	bindResource := func(index int, v *allocation) {
		if v == nil {
			return
		}
		buf := bufferBinding{Resource: v.native.(*object), Size: v.size}
		bdesc := bindingDesc{Type: 1, Desc: &buf}
		if index == 9 {
			call(d.batchTable, index, 1, uintptr(unsafe.Pointer(&bdesc)))
		} else {
			call(d.batchTable, index, uintptr(unsafe.Pointer(&bdesc)))
		}
	}
	bindResource(10, initTemp)
	bindResource(9, op.persistent)
	if err := d.recordInit(initializer, initTemp, op.persistent); err != nil {
		return nil, err
	}
	return op, nil
}

func (d *nativeDriver) recordInit(initializer *object, resources ...*allocation) error {
	var used []*allocation
	for _, v := range resources {
		if v == nil {
			continue
		}
		found := false
		for _, old := range used {
			found = found || old == v
		}
		if !found {
			used = append(used, v)
		}
	}
	return d.submit(func() error {
		for _, v := range used {
			d.transition(v, 0, stateUAV)
		}
		call(d.command, 28, 1, uintptr(unsafe.Pointer(&d.batchHeap)))
		call(d.recorder, 8, uintptr(unsafe.Pointer(d.command)), uintptr(unsafe.Pointer(initializer)), uintptr(unsafe.Pointer(d.batchTable)))
		d.uavBarrier()
		for _, v := range used {
			d.transition(v, stateUAV, 0)
		}
		return nil
	})
}

func (d *nativeDriver) hasQueued() bool        { return len(d.queued) != 0 }
func (d *nativeDriver) discard()               { d.queued = nil; d.queuedRegions = 0 }
func (d *nativeDriver) setFlushHook(fn func()) { d.onFlush = fn }

func (d *nativeDriver) enqueue(step queuedStep) error {
	if step.regions > 0 && d.queuedRegions > 0 && d.queuedRegions+step.regions > maxCopyRegions {
		if err := d.flush(); err != nil {
			return err
		}
	}
	d.queued = append(d.queued, step)
	d.queuedRegions += step.regions
	if len(d.queued) >= maxQueuedOps {
		return d.flush()
	}
	return nil
}

func (d *nativeDriver) execute(key opKey, out *allocation, inputs ...*allocation) error {
	switch key.kind {
	case opClamp, opPow, opLogSoftmax:
		if err := d.initUnaryShader(key.kind); err != nil {
			return err
		}
		return d.enqueue(queuedStep{kind: stepUnaryShader, key: key, out: out, src: inputs[0]})
	}
	bindings := make([]resourceBinding, len(inputs))
	for i, v := range inputs {
		bindings[i].memory = v
	}
	return d.executeAt(key, out, bindings...)
}

func (d *nativeDriver) executeAt(key opKey, out *allocation, inputs ...resourceBinding) error {
	copied := append([]resourceBinding(nil), inputs...)
	return d.enqueue(queuedStep{kind: stepDispatch, key: key, out: out, inputs: copied})
}

func (d *nativeDriver) executeBatch(steps []dispatch) error {
	for _, step := range steps {
		bindings := make([]resourceBinding, len(step.inputs))
		for i, v := range step.inputs {
			bindings[i].memory = v
		}
		if err := d.enqueue(queuedStep{kind: stepDispatch, key: step.key, out: step.out, inputs: bindings}); err != nil {
			return err
		}
	}
	return nil
}

func (d *nativeDriver) flush() error {
	if len(d.queued) == 0 {
		return nil
	}
	steps := d.queued
	d.queued = nil
	d.queuedRegions = 0
	defer func() {
		if d.onFlush != nil {
			d.onFlush()
		}
	}()
	for start := 0; start < len(steps); {
		pinned := make(map[*compiled]bool)
		ownedByKey := make(map[opKey]*compiled)
		var owned []*compiled
		resolved := make([]*compiled, 0, len(steps)-start)
		var usedDescs uint32
		end := start
		var prepErr error
		for end < len(steps) {
			step := steps[end]
			if step.kind != stepDispatch {
				resolved = append(resolved, nil)
				end++
				continue
			}
			op, reused := ownedByKey[step.key]
			var isOwned bool
			if !reused {
				var err error
				op, isOwned, err = d.acquireOperator(step.key, pinned)
				if err != nil {
					prepErr = err
					break
				}
			}
			need := op.execDescriptors
			if usedDescs > 0 && usedDescs+need > d.batchHeapSize {
				if isOwned {
					d.closeOperator(op)
				}
				break
			}
			if need > d.batchHeapSize {
				if err := d.ensureBatchHeap(need); err != nil {
					if isOwned {
						d.closeOperator(op)
					}
					prepErr = err
					break
				}
			}
			if isOwned {
				owned = append(owned, op)
				ownedByKey[step.key] = op
			}
			pinned[op] = true
			usedDescs += need
			resolved = append(resolved, op)
			end++
		}
		if prepErr != nil {
			for _, op := range owned {
				d.closeOperator(op)
			}
			return prepErr
		}
		err := d.submit(func() error {
			state := make(map[*allocation]uint32)
			ensureState := func(v *allocation, want uint32) {
				if v == nil || v.heap != heapDefault {
					return
				}
				if cur := state[v]; cur != want {
					d.transition(v, cur, want)
					state[v] = want
				}
			}
			heapBound := false
			var cursor uint32
			for idx, step := range steps[start:end] {
				switch step.kind {
				case stepDispatch:
					op := resolved[idx]
					if !heapBound {
						call(d.command, 28, 1, uintptr(unsafe.Pointer(&d.batchHeap)))
						heapBound = true
					}
					ensureState(step.out, stateUAV)
					ensureState(op.temporary, stateUAV)
					ensureState(op.persistent, stateUAV)
					for _, in := range step.inputs {
						ensureState(in.memory, stateUAV)
					}
					bdesc := bindingTableDesc{
						Dispatchable: op.operator,
						CPU:          d.batchCPU + uint64(cursor)*d.descSize,
						GPU:          d.batchGPU + uint64(cursor)*d.descSize,
						Count:        op.execDescriptors,
					}
					cursor += op.execDescriptors
					if err := hresult("Reset(batch table)", call(d.batchTable, 12, uintptr(unsafe.Pointer(&bdesc)))); err != nil {
						return err
					}
					d.bindTable(d.batchTable, op, step.key, step.out, step.inputs)
					call(d.recorder, 8, uintptr(unsafe.Pointer(d.command)), uintptr(unsafe.Pointer(op.operator)), uintptr(unsafe.Pointer(d.batchTable)))
					d.uavBarrier()
				case stepUnaryShader:
					ensureState(step.src, stateUAV)
					ensureState(step.out, stateUAV)
					d.recordUnaryShader(step.key, step.out, step.src)
					d.uavBarrier()
				case stepCopy:
					ensureState(step.src, stateCopySource)
					ensureState(step.out, stateCopyDest)
					call(d.command, 15, uintptr(unsafe.Pointer(step.out.native.(*object))), 0, uintptr(unsafe.Pointer(step.src.native.(*object))), 0, uintptr(step.size))
				case stepCopyView:
					ensureState(step.src, stateCopySource)
					ensureState(step.out, stateCopyDest)
					d.recordCopyView(step.out, step.src, step.view, step.run, step.scatter)
				}
			}
			for v, cur := range state {
				if cur != 0 {
					d.transition(v, cur, 0)
				}
			}
			return nil
		})
		for _, op := range owned {
			d.closeOperator(op)
		}
		if err != nil {
			return err
		}
		start = end
	}
	return nil
}

// owned means the caller must keep the operator alive until its fence completes
// and then close it. Cache entries remain bounded even when every potential
// eviction candidate is pinned.
func (d *nativeDriver) acquireOperator(key opKey, pinned map[*compiled]bool) (_ *compiled, owned bool, err error) {
	op := d.operators[key]
	if op == nil {
		op, err = d.compile(key)
		if err != nil {
			return nil, false, err
		}
		var bytes uint64
		if op.temporary != nil {
			bytes += op.temporary.allocated
		}
		if op.persistent != nil {
			bytes += op.persistent.allocated
		}
		if bytes > operatorCacheBytes {
			return op, true, nil
		} else {
			for {
				stats := d.operatorStats()
				if len(d.order) < maxCachedOperators && stats.TemporaryBytes+stats.PersistentBytes+bytes <= operatorCacheBytes {
					break
				}
				index := -1
				for i, k := range d.order {
					if !pinned[d.operators[k]] {
						index = i
						break
					}
				}
				if index < 0 {
					return op, true, nil
				}
				old := d.order[index]
				d.closeOperator(d.operators[old])
				delete(d.operators, old)
				d.order = append(d.order[:index], d.order[index+1:]...)
			}
			if d.operators == nil {
				d.operators = make(map[opKey]*compiled)
			}
			d.operators[key] = op
			d.order = append(d.order, key)
		}
	}
	return op, false, nil
}

func (d *nativeDriver) bindTable(table *object, op *compiled, key opKey, out *allocation, inputs []resourceBinding) {
	if key.kind == opScalar || key.kind == opIdentity {
		inputs = inputs[:1]
	}
	if key.kind == opGEMM {
		inputs = append(append([]resourceBinding(nil), inputs...), resourceBinding{})
	}
	if key.kind == opNorm {
		inputs = append(append([]resourceBinding(nil), inputs...), resourceBinding{}, resourceBinding{})
	}
	buffers := make([]bufferBinding, len(inputs))
	descs := make([]bindingDesc, len(inputs))
	for i, v := range inputs {
		if v.memory != nil {
			buffers[i] = bufferBinding{Resource: v.memory.native.(*object), Offset: v.offset, Size: v.memory.size - v.offset}
			descs[i] = bindingDesc{Type: 1, Desc: &buffers[i]}
		}
	}
	call(table, 8, uintptr(len(descs)), uintptr(unsafe.Pointer(&descs[0])))
	output := bufferBinding{Resource: out.native.(*object), Size: out.size}
	outputDescs := [2]bindingDesc{{Type: 1, Desc: &output}}
	outputCount := uintptr(1)
	if key.kind == opMaxPool2 {
		outputCount = 2
	}
	call(table, 9, outputCount, uintptr(unsafe.Pointer(&outputDescs[0])))
	if op.temporary != nil {
		tb := bufferBinding{Resource: op.temporary.native.(*object), Size: op.temporary.size}
		td := bindingDesc{Type: 1, Desc: &tb}
		call(table, 10, uintptr(unsafe.Pointer(&td)))
	}
	if op.persistent != nil {
		pb := bufferBinding{Resource: op.persistent.native.(*object), Size: op.persistent.size}
		pd := bindingDesc{Type: 1, Desc: &pb}
		call(table, 11, uintptr(unsafe.Pointer(&pd)))
	}
}
