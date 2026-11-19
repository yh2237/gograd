package directml

import (
	"math"

	"github.com/yh2237/gograd/gpu"
)

var (
	_ gpu.BinaryOpBackend     = (*Backend)(nil)
	_ gpu.ViewBinaryOpBackend = (*Backend)(nil)
	_ gpu.ScalarOpBackend     = (*Backend)(nil)
	_ gpu.UnaryOpBackend      = (*Backend)(nil)
	_ gpu.LogSoftmaxBackend   = (*Backend)(nil)
	_ gpu.ReduceBackend       = (*Backend)(nil)
	_ gpu.VisionBackend       = (*Backend)(nil)
	_ gpu.BatchNormBackend    = (*Backend)(nil)
)

const (
	sdkOp        uint32 = 256
	opAbs               = sdkOp + 2
	opClip              = sdkOp + 8
	opCos               = sdkOp + 9
	opDiv               = sdkOp + 10
	opExp               = sdkOp + 11
	opLog               = sdkOp + 13
	opMax               = sdkOp + 21
	opMin               = sdkOp + 23
	opPow               = sdkOp + 26
	opSin               = sdkOp + 28
	opSqrt              = sdkOp + 29
	opSub               = sdkOp + 30
	opLogSoftmax        = sdkOp + 41
	opReLU              = sdkOp + 44
	opSigmoid           = sdkOp + 47
	opSoftplus          = sdkOp + 49
	opConv2d            = sdkOp + 53
	opReduce            = sdkOp + 55
	opAvgPool           = sdkOp + 56
	opMaxPool           = sdkOp + 58
	opBatchNorm         = sdkOp + 72
	opErf               = sdkOp + 81
	opMaxPool2          = sdkOp + 110
)

var binaryKinds = [...]uint32{opSub, opDiv, opMax, opMin}
var unaryKinds = [...]uint32{
	opExp, opLog, opAbs, opTanh, opReLU, opSin,
	opCos, opSqrt, opSigmoid, opSoftplus, opErf,
}

func (b *Backend) BinaryOp(op gpu.BinaryOp, out, a, right *gpu.Buffer, n int) error {
	if int(op) >= len(binaryKinds) {
		return gpu.ErrInvalidArgument
	}
	return b.pointwise(out, a, right, n, binaryKinds[op], 0)
}

func (b *Backend) BinaryOpViews(op gpu.BinaryOp, out *gpu.Buffer, a, right gpu.View) error {
	if int(op) >= len(binaryKinds) {
		return gpu.ErrInvalidArgument
	}
	return b.binaryViews(out, a, right, binaryKinds[op])
}

func (b *Backend) UnaryOp(op gpu.UnaryOp, out, a *gpu.Buffer, n int) error {
	if int(op) >= len(unaryKinds) {
		return gpu.ErrInvalidArgument
	}
	return b.unary(out, a, n, opKey{kind: unaryKinds[op], n: n})
}

func (b *Backend) Clamp(out, a *gpu.Buffer, n int, lo, hi float32) error {
	if lo > hi || math.IsNaN(float64(lo)) || math.IsNaN(float64(hi)) {
		return gpu.ErrUnsupported
	}
	return b.unary(out, a, n, opKey{kind: opClip, n: n, scalar: math.Float32bits(lo), bias: math.Float32bits(hi)})
}

func (b *Backend) LogSoftmax(out, a *gpu.Buffer, rows, cols int) error {
	return b.softmax(out, a, rows, cols, opLogSoftmax)
}

func (b *Backend) ScalarOp(op gpu.ScalarOp, out, a *gpu.Buffer, n int, value float32) error {
	key := opKey{kind: opScalar, n: n, scalar: math.Float32bits(1)}
	switch op {
	case gpu.ScalarSub:
		value = -value
		fallthrough
	case gpu.ScalarAdd:
		key.bias = math.Float32bits(value)
	case gpu.ScalarPow:
		key.kind = opPow
		key.scalar = math.Float32bits(value)
	case gpu.ScalarDiv:
		scale := 1 / value
		magnitude := math.Abs(float64(scale))
		if math.Float32bits(value)&0x7fffff != 0 || !(magnitude >= 0x1p-126 && magnitude <= math.MaxFloat32) {
			return b.scalarDivide(out, a, n, value)
		}
		key.scalar = math.Float32bits(scale)
		key.bias = 0x80000000
	default:
		return gpu.ErrInvalidArgument
	}
	return b.unary(out, a, n, key)
}

func (b *Backend) scalarDivide(out, a *gpu.Buffer, n int, value float32) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, n, bufferArg{a, n})
	if err != nil || n == 0 {
		return err
	}
	// DirectML can flush subnormal operands to zero during division.
	if value != 0 && math.Abs(float64(value)) < 0x1p-126 {
		return gpu.ErrUnsupported
	}
	scalar, err := b.take(4, heapDefault)
	if err != nil {
		return err
	}
	defer b.put(scalar)
	if err := b.fillLocked(scalar, 1, value); err != nil {
		return err
	}
	layout := tensorLayout{sizes: [4]uint32{1, 1, 1, uint32(n)}}
	return b.fail(b.driver.execute(opKey{kind: opDiv, n: n, bLayout: layout}, y, inputs[0], scalar))
}

type bufferArg struct {
	buffer *gpu.Buffer
	size   int
}

// opBuffers validates all bindings before any work is queued. A negative size
// marks an absent optional input.
func (b *Backend) opBuffers(out *gpu.Buffer, n int, args ...bufferArg) (*allocation, []*allocation, error) {
	y, err := b.native(out, n)
	if err != nil {
		return nil, nil, err
	}
	inputs := make([]*allocation, len(args))
	for i, arg := range args {
		if arg.size < 0 {
			continue
		}
		x, err := b.native(arg.buffer, arg.size)
		if err != nil {
			return nil, nil, err
		}
		if x == y {
			return nil, nil, gpu.ErrInvalidArgument
		}
		inputs[i] = x
	}
	return y, inputs, nil
}

func optionalBuffer(buffer *gpu.Buffer, n int) bufferArg {
	if buffer == nil {
		n = -1
	}
	return bufferArg{buffer, n}
}

func (b *Backend) fillLocked(out *allocation, n int, value float32) error {
	bytes := uint64(n) * 4
	staging, err := b.take(bytes, heapUpload)
	if err != nil {
		return err
	}
	defer b.put(staging)
	var values []float32
	if math.Float32bits(value) != 0 {
		values = make([]float32, n)
		for i := range values {
			values[i] = value
		}
	}
	if err := b.fail(b.driver.write(staging, values)); err != nil {
		return err
	}
	return b.fail(b.driver.copy(out, staging, bytes))
}

type reduceDim struct {
	size    int
	reduced bool
}

func mergeReduceDims(dims []reduceDim) []reduceDim {
	merged := dims[:0]
	for _, d := range dims {
		if d.size == 1 {
			continue
		}
		if len(merged) > 0 && merged[len(merged)-1].reduced == d.reduced {
			merged[len(merged)-1].size *= d.size
		} else {
			merged = append(merged, d)
		}
	}
	return merged
}

func reduceKey(function uint32, dims []reduceDim, n int) opKey {
	in, out := [4]uint32{1, 1, 1, 1}, [4]uint32{1, 1, 1, 1}
	var mask uint32
	for i, d := range dims {
		axis := 4 - len(dims) + i
		in[axis] = uint32(d.size)
		if d.reduced {
			mask |= 1 << axis
		} else {
			out[axis] = in[axis]
		}
	}
	return opKey{kind: opReduce, n: n, scalar: function, bias: mask, aLayout: layout4(in), yLayout: layout4(out)}
}

func layout4(s [4]uint32) tensorLayout { return compactLayout(s[0], s[1], s[2], s[3]) }

func (b *Backend) Reduce(op gpu.ReduceOp, out, a *gpu.Buffer, shape, axes []int) error {
	var function uint32
	switch op {
	case gpu.ReduceSum:
		function = 10
	case gpu.ReduceMean:
		function = 2
	default:
		return gpu.ErrInvalidArgument
	}
	if len(shape) > 6 {
		return gpu.ErrInvalidArgument
	}
	inN, err := gpu.Elements(shape...)
	if err != nil {
		return err
	}
	dims := make([]reduceDim, len(shape))
	for i, s := range shape {
		dims[i].size = s
	}
	for _, axis := range axes {
		if axis < 0 {
			axis += len(shape)
		}
		if axis < 0 || axis >= len(shape) || dims[axis].reduced {
			return gpu.ErrInvalidArgument
		}
		dims[axis].reduced = true
	}
	var kept []int
	for _, d := range dims {
		if !d.reduced {
			kept = append(kept, d.size)
		}
	}
	outN, err := gpu.Elements(kept...)
	if err != nil {
		return err
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, outN, bufferArg{a, inN})
	if err != nil || outN == 0 {
		return err
	}
	x := inputs[0]
	if inN == 0 {
		var value float32
		if op == gpu.ReduceMean {
			value = math.Float32frombits(0x7fc00000)
		}
		return b.fillLocked(y, outN, value)
	}
	if inN == outN {
		return b.fail(b.driver.copy(y, x, uint64(inN)*4))
	}
	dims = mergeReduceDims(dims)
	if len(dims) > 4 {
		// Remove an interior reduced run so its two kept neighbors merge.
		axis := 1
		if !dims[axis].reduced {
			axis++
		}
		prefix := 1
		for _, d := range dims[:axis] {
			prefix *= d.size
		}
		run := dims[axis]
		middleN := inN / run.size
		temp, err := b.take(uint64(middleN)*4, heapDefault)
		if err != nil {
			return err
		}
		defer b.put(temp)
		first := reduceKey(function, []reduceDim{{prefix, false}, run, {middleN / prefix, false}}, inN)
		if err := b.driver.execute(first, temp, x); err != nil {
			return b.fail(err)
		}
		dims[axis].size = 1
		dims = mergeReduceDims(dims)
		x, inN = temp, middleN
	}
	return b.fail(b.driver.execute(reduceKey(function, dims, inN), y, x))
}

func (b *Backend) Conv2d(out, input, weight, bias *gpu.Buffer, o gpu.Conv2dOptions) error {
	oh, ow, err := o.OutputShape()
	if err != nil {
		return err
	}
	xSize, err := gpu.Elements(o.Batch, o.InChannels, o.Height, o.Width)
	if err != nil {
		return err
	}
	wSize, err := gpu.Elements(o.OutChannels, o.InChannels/o.Groups, o.KernelH, o.KernelW)
	if err != nil {
		return err
	}
	ySize, err := gpu.Elements(o.Batch, o.OutChannels, oh, ow)
	if err != nil {
		return err
	}
	key := opKey{kind: opConv2d, n: ySize, hasBias: bias != nil, scalar: uint32(o.Groups),
		aLayout: compactLayout(uint32(o.Batch), uint32(o.InChannels), uint32(o.Height), uint32(o.Width)),
		bLayout: compactLayout(uint32(o.OutChannels), uint32(o.InChannels/o.Groups), uint32(o.KernelH), uint32(o.KernelW)),
		yLayout: compactLayout(uint32(o.Batch), uint32(o.OutChannels), uint32(oh), uint32(ow)),
		spatial: spatialKey{stride: [2]uint32{uint32(o.StrideH), uint32(o.StrideW)}, dilation: [2]uint32{uint32(o.DilationH), uint32(o.DilationW)}, start: [2]uint32{uint32(o.PadH), uint32(o.PadW)}, end: [2]uint32{uint32(o.PadH), uint32(o.PadW)}}}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, ySize, bufferArg{input, xSize}, bufferArg{weight, wSize}, optionalBuffer(bias, o.OutChannels))
	if err != nil {
		return err
	}
	// WARP can write zeros when a single-position dilated window spans an
	// input whose spatial extent on that axis equals the undilated kernel.
	if oh == 1 && ow == 1 &&
		((o.DilationH > 1 && o.Height == o.KernelH) || (o.DilationW > 1 && o.Width == o.KernelW)) {
		info := b.driver.info()
		if info.VendorID == 0x1414 && info.DeviceID == 0x008c {
			return gpu.ErrUnsupported
		}
	}
	return b.fail(b.driver.execute(key, y, inputs...))
}

func (b *Backend) MaxPool2d(out, x *gpu.Buffer, o gpu.Pool2dOptions) error {
	return b.pool2d(out, x, o, false)
}
func (b *Backend) AvgPool2d(out, x *gpu.Buffer, o gpu.Pool2dOptions) error {
	return b.pool2d(out, x, o, true)
}

func validPoolAxis(input, kernel, stride, start, end, dilation, output int) bool {
	for _, d := range []int{input, kernel, stride, dilation, output} {
		if d < 1 || d > math.MaxInt32 {
			return false
		}
	}
	if start < 0 || start > math.MaxInt32 || end < 0 || end > math.MaxInt32 {
		return false
	}
	padded := int64(input) + int64(start) + int64(end)
	span := int64(dilation)*int64(kernel-1) + 1
	if padded > math.MaxInt32 || span > math.MaxInt32 || span > padded {
		return false
	}
	return (padded-span)/int64(stride)+1 == int64(output)
}

func (b *Backend) pool2d(out, x *gpu.Buffer, o gpu.Pool2dOptions, avg bool) error {
	if o.Batch < 1 || o.Channels < 1 ||
		!validPoolAxis(o.Height, o.KernelH, o.StrideH, o.PadH, o.EndPadH, o.DilationH, o.OutH) ||
		!validPoolAxis(o.Width, o.KernelW, o.StrideW, o.PadW, o.EndPadW, o.DilationW, o.OutW) {
		return gpu.ErrInvalidArgument
	}
	xSize, err := gpu.Elements(o.Batch, o.Channels, o.Height, o.Width)
	if err != nil {
		return err
	}
	ySize, err := gpu.Elements(o.Batch, o.Channels, o.OutH, o.OutW)
	if err != nil {
		return err
	}
	key := opKey{kind: opMaxPool, n: ySize,
		aLayout: compactLayout(uint32(o.Batch), uint32(o.Channels), uint32(o.Height), uint32(o.Width)),
		yLayout: compactLayout(uint32(o.Batch), uint32(o.Channels), uint32(o.OutH), uint32(o.OutW)),
		spatial: spatialKey{stride: [2]uint32{uint32(o.StrideH), uint32(o.StrideW)}, dilation: [2]uint32{uint32(o.DilationH), uint32(o.DilationW)}, start: [2]uint32{uint32(o.PadH), uint32(o.PadW)}, end: [2]uint32{uint32(o.EndPadH), uint32(o.EndPadW)}, window: [2]uint32{uint32(o.KernelH), uint32(o.KernelW)}}}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, ySize, bufferArg{x, xSize})
	if err != nil {
		return err
	}
	if o.DilationH != 1 || o.DilationW != 1 {
		if avg || b.driver.info().FeatureLevel < 0x2100 {
			return gpu.ErrUnsupported
		}
		key.kind = opMaxPool2
	} else if avg {
		key.kind = opAvgPool
		if o.IncludePad {
			key.scalar = 1
		}
	}
	return b.fail(b.driver.execute(key, y, inputs...))
}

func (b *Backend) BatchNormEval(out, x, weight, bias, mean, variance *gpu.Buffer, outer, channels, inner int, eps float32) error {
	if outer < 1 || channels < 1 || inner < 1 || !validEps(eps) {
		return gpu.ErrInvalidArgument
	}
	n, err := gpu.Elements(outer, channels, inner)
	if err != nil {
		return err
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, n, bufferArg{x, n}, bufferArg{mean, channels}, bufferArg{variance, channels}, optionalBuffer(weight, channels), optionalBuffer(bias, channels))
	if err != nil {
		return err
	}
	for i, value := range [...]float32{1, 0} {
		if inputs[3+i] != nil {
			continue
		}
		temp, err := b.take(uint64(channels)*4, heapDefault)
		if err != nil {
			return err
		}
		defer b.put(temp)
		if err := b.fillLocked(temp, channels, value); err != nil {
			return err
		}
		inputs[3+i] = temp
	}
	layout := compactLayout(uint32(outer), uint32(channels), 1, uint32(inner))
	key := opKey{kind: opBatchNorm, n: n, scalar: math.Float32bits(eps), aLayout: layout, yLayout: layout,
		bLayout: compactLayout(1, uint32(channels), 1, 1)}
	return b.fail(b.driver.execute(key, y, inputs...))
}
