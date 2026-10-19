//go:build windows && amd64

package directml

import (
	"math"
	"unsafe"
)

type compiled struct {
	operator, heap, table *object
	temporary, persistent *allocation
	desc                  bindingTableDesc
}

func (d *nativeDriver) closeOperator(op *compiled) {
	release(op.table)
	release(op.heap)
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
	heapDesc := descriptorHeapDesc{Count: max(1, max(initProps.Descriptors, execProps.Descriptors)), Flags: 1}
	if err := hresult("CreateDescriptorHeap", call(d.device, 14, uintptr(unsafe.Pointer(&heapDesc)), uintptr(unsafe.Pointer(iidHeap)), uintptr(unsafe.Pointer(&op.heap)))); err != nil {
		return nil, err
	}
	call(op.heap, 9, uintptr(unsafe.Pointer(&op.desc.CPU)))
	call(op.heap, 10, uintptr(unsafe.Pointer(&op.desc.GPU)))
	op.desc.Dispatchable, op.desc.Count = initializer, heapDesc.Count
	if err := hresult("CreateBindingTable", call(d.dml, 12, uintptr(unsafe.Pointer(&op.desc)), uintptr(unsafe.Pointer(iidBindingTable)), uintptr(unsafe.Pointer(&op.table)))); err != nil {
		return nil, err
	}
	if size := max(initProps.Temporary, execProps.Temporary); size != 0 {
		op.temporary, err = d.alloc(size, heapDefault)
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
	bind := func(index int, v *allocation) {
		if v == nil {
			return
		}
		buf := bufferBinding{Resource: v.native.(*object), Size: v.size}
		desc := bindingDesc{Type: 1, Desc: &buf}
		if index == 9 {
			call(op.table, index, 1, uintptr(unsafe.Pointer(&desc)))
		} else {
			call(op.table, index, uintptr(unsafe.Pointer(&desc)))
		}
	}
	if initProps.Temporary != 0 {
		bind(10, op.temporary)
	}
	bind(9, op.persistent) // Initializer output initializes the persistent resource.
	if err := d.recordDispatch(op, initializer, op.temporary, op.persistent); err != nil {
		return nil, err
	}
	op.desc.Dispatchable = op.operator
	if err := hresult("Reset(binding table)", call(op.table, 12, uintptr(unsafe.Pointer(&op.desc)))); err != nil {
		return nil, err
	}
	if execProps.Temporary != 0 {
		bind(10, op.temporary)
	}
	bind(11, op.persistent)
	return op, nil
}

func (d *nativeDriver) recordDispatch(op *compiled, dispatchable *object, resources ...*allocation) error {
	// All default buffers are explicitly returned to COMMON before submission
	// completes. That avoids relying on implicit promotion/decay between lists.
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
	return d.submit(func() {
		for _, v := range used {
			d.transition(v, 0, stateUAV)
		}
		call(d.command, 28, 1, uintptr(unsafe.Pointer(&op.heap)))
		call(d.recorder, 8, uintptr(unsafe.Pointer(d.command)), uintptr(unsafe.Pointer(dispatchable)), uintptr(unsafe.Pointer(op.table)))
		d.uavBarrier()
		for _, v := range used {
			d.transition(v, stateUAV, 0)
		}
	})
}

func (d *nativeDriver) execute(key opKey, out *allocation, inputs ...*allocation) error {
	bindings := make([]resourceBinding, len(inputs))
	for i, v := range inputs {
		bindings[i].memory = v
	}
	return d.executeAt(key, out, bindings...)
}

func (d *nativeDriver) executeAt(key opKey, out *allocation, inputs ...resourceBinding) error {
	op := d.operators[key]
	if op == nil {
		var err error
		op, err = d.compile(key)
		if err != nil {
			return err
		}
		var bytes uint64
		if op.temporary != nil {
			bytes += op.temporary.allocated
		}
		if op.persistent != nil {
			bytes += op.persistent.allocated
		}
		if bytes > operatorCacheBytes {
			defer d.closeOperator(op)
		} else {
			for len(d.order) > 0 {
				stats := d.operatorStats()
				if len(d.order) < maxCachedOperators && stats.TemporaryBytes+stats.PersistentBytes+bytes <= operatorCacheBytes {
					break
				}
				old := d.order[0]
				d.closeOperator(d.operators[old])
				delete(d.operators, old)
				d.order = d.order[1:]
			}
			if d.operators == nil {
				d.operators = make(map[opKey]*compiled)
			}
			d.operators[key] = op
			d.order = append(d.order, key)
		}
	}
	if key.kind == opScalar || key.kind == opIdentity {
		inputs = inputs[:1]
	}
	// GEMM has a third, optional C input. Bind NONE when Beta is zero.
	if key.kind == opGEMM {
		inputs = append(inputs, resourceBinding{})
	}
	if key.kind == opNorm {
		inputs = append(inputs, resourceBinding{}, resourceBinding{})
	}
	buffers := make([]bufferBinding, len(inputs))
	descs := make([]bindingDesc, len(inputs))
	for i, v := range inputs {
		if v.memory != nil {
			buffers[i] = bufferBinding{Resource: v.memory.native.(*object), Offset: v.offset, Size: v.memory.size - v.offset}
			descs[i] = bindingDesc{Type: 1, Desc: &buffers[i]}
		}
	}
	call(op.table, 8, uintptr(len(descs)), uintptr(unsafe.Pointer(&descs[0])))
	output := bufferBinding{Resource: out.native.(*object), Size: out.size}
	outputDesc := bindingDesc{Type: 1, Desc: &output}
	call(op.table, 9, 1, uintptr(unsafe.Pointer(&outputDesc)))
	resources := []*allocation{out, op.temporary, op.persistent}
	for _, v := range inputs {
		resources = append(resources, v.memory)
	}
	return d.recordDispatch(op, op.operator, resources...)
}
