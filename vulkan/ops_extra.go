package vulkan

import (
	"fmt"
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

type ndReduceParams struct {
	OutN, ReduceN, OutRank, RedRank, Op uint32
	OutShape, OutInStrides              [6]uint32
	RedShape, RedInStrides              [6]uint32
}

type visionParams struct {
	Kind, N          uint32
	D0, D1, D2, D3   uint32
	D4, D5, D6, D7   uint32
	D8, D9, D10, D11 uint32
	D12, D13         uint32
	F0               float32
}

type conv2dBatch struct {
	out, input, weight, bias, col                          *block
	outCount, inputCount, weightCount, biasCount, colCount uint32
	im2col, biasOp                                         visionParams
	gemm                                                   gemmParams
	variant                                                int
	hasBias                                                bool
}

var (
	binaryOpCodes   = [...]uint32{7, 8, 9, 10}
	binaryViewCodes = [...]uint32{2, 3, 4, 5}
	scalarOpCodes   = [...]uint32{11, 12, 13, 14}
	unaryOpCodes    = [...]uint32{15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
)

func (b *Backend) BinaryOp(op gpu.BinaryOp, out, a, right *gpu.Buffer, n int) error {
	if int(op) >= len(binaryOpCodes) {
		return gpu.ErrInvalidArgument
	}
	return b.elementwise(out, a, right, n, binaryOpCodes[op], 0)
}

func (b *Backend) BinaryOpViews(op gpu.BinaryOp, out *gpu.Buffer, a, right gpu.View) error {
	if int(op) >= len(binaryViewCodes) {
		return gpu.ErrInvalidArgument
	}
	return b.binaryViews(out, a, right, binaryViewCodes[op])
}

func (b *Backend) ScalarOp(op gpu.ScalarOp, out, a *gpu.Buffer, n int, value float32) error {
	if int(op) >= len(scalarOpCodes) {
		return gpu.ErrInvalidArgument
	}
	return b.elementwise(out, a, a, n, scalarOpCodes[op], value)
}

func (b *Backend) UnaryOp(op gpu.UnaryOp, out, a *gpu.Buffer, n int) error {
	if int(op) >= len(unaryOpCodes) {
		return gpu.ErrInvalidArgument
	}
	return b.elementwise(out, a, a, n, unaryOpCodes[op], 0)
}

func (b *Backend) Clamp(out, a *gpu.Buffer, n int, lo, hi float32) error {
	return b.elementwise2(out, a, a, n, 26, lo, hi)
}

func (b *Backend) LogSoftmax(out, a *gpu.Buffer, rows, cols int) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 {
		return fmt.Errorf("%w: empty log_softmax axis", gpu.ErrInvalidArgument)
	}
	return b.reduction(out, a, nil, nil, n, 0, reduceParams{Rows: uint32(rows), Cols: uint32(cols), Kind: 3})
}

func (b *Backend) Reduce(op gpu.ReduceOp, out, a *gpu.Buffer, shape, axes []int) error {
	var opCode uint32
	switch op {
	case gpu.ReduceSum:
		opCode = 0
	case gpu.ReduceMean:
		opCode = 1
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
	reduced := make([]bool, len(shape))
	for _, ax := range axes {
		if ax < 0 {
			ax += len(shape)
		}
		if ax < 0 || ax >= len(shape) || reduced[ax] {
			return gpu.ErrInvalidArgument
		}
		reduced[ax] = true
	}
	var kept []int
	for i, d := range shape {
		if !reduced[i] {
			kept = append(kept, d)
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
	x, err := b.native(a, inN)
	if err != nil {
		return err
	}
	y, err := b.native(out, outN)
	if err != nil {
		return err
	}
	if y == x {
		return fmt.Errorf("%w: reduction output aliases input", gpu.ErrInvalidArgument)
	}
	if outN == 0 {
		return nil
	}
	if inN == 0 {
		if op == gpu.ReduceSum {
			return b.fail(b.driver.zero(y, uint64(outN)*4))
		}
		if _, _, err := dispatchGrid(uint32(outN), b.driver.info().MaxWorkGroupCount); err != nil {
			return err
		}
		return b.fail(b.driver.pointwise(y, y, y, uint32(outN), 27, math.Float32frombits(0x7fc00000), 0))
	}
	if inN == outN {
		return b.fail(b.driver.copy(y, x, uint64(inN)*4))
	}
	if _, _, err := groupGrid(uint64(outN), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	inStrides := make([]uint32, len(shape))
	stride := uint32(1)
	for i := len(shape) - 1; i >= 0; i-- {
		inStrides[i] = stride
		stride *= uint32(shape[i])
	}
	p := ndReduceParams{OutN: uint32(outN), ReduceN: uint32(inN / outN), Op: opCode}
	for i, d := range shape {
		if reduced[i] {
			p.RedShape[p.RedRank] = uint32(d)
			p.RedInStrides[p.RedRank] = inStrides[i]
			p.RedRank++
		} else {
			p.OutShape[p.OutRank] = uint32(d)
			p.OutInStrides[p.OutRank] = inStrides[i]
			p.OutRank++
		}
	}
	return b.fail(b.driver.ndReduce(y, x, uint32(outN), uint32(inN), p))
}

func (b *Backend) Conv2d(out, input, weight, bias *gpu.Buffer, o gpu.Conv2dOptions) error {
	oh, ow, err := o.OutputShape()
	if err != nil {
		return err
	}
	ciPerG := o.InChannels / o.Groups
	coPerG := o.OutChannels / o.Groups
	xSize, err := gpu.Elements(o.Batch, o.InChannels, o.Height, o.Width)
	if err != nil {
		return err
	}
	wSize, err := gpu.Elements(o.OutChannels, ciPerG, o.KernelH, o.KernelW)
	if err != nil {
		return err
	}
	ySize, err := gpu.Elements(o.Batch, o.OutChannels, oh, ow)
	if err != nil {
		return err
	}
	depth, err := gpu.Elements(ciPerG, o.KernelH, o.KernelW)
	if err != nil {
		return err
	}
	spatial, err := gpu.Elements(oh, ow)
	if err != nil {
		return err
	}
	colSize, err := gpu.Elements(o.Batch, o.Groups, depth, spatial)
	if err != nil {
		return err
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(input, xSize)
	if err != nil {
		return err
	}
	w, err := b.native(weight, wSize)
	if err != nil {
		return err
	}
	y, err := b.native(out, ySize)
	if err != nil {
		return err
	}
	v := x
	biasCount := xSize
	if bias != nil {
		v, err = b.native(bias, o.OutChannels)
		if err != nil {
			return err
		}
		biasCount = o.OutChannels
	}
	if y == x || y == w || (bias != nil && y == v) {
		return fmt.Errorf("%w: Conv2d output aliases input", gpu.ErrInvalidArgument)
	}
	if uint64(colSize)*4 > b.driver.info().MaxStorageBufferBytes {
		return fmt.Errorf("%w: Conv2d im2col exceeds storage buffer range", gpu.ErrUnsupported)
	}
	p, variant := matrixParams(
		gpu.View{Shape: []int{1, o.Groups, coPerG, depth}, Strides: []int{0, coPerG * depth, depth, 1}},
		gpu.View{Shape: []int{o.Batch, o.Groups, depth, spatial}, Strides: []int{o.Groups * depth * spatial, depth * spatial, spatial, 1}},
		[]int{o.Batch, o.Groups, coPerG, spatial},
	)
	limits := b.driver.info().MaxWorkGroupCount
	if _, _, err := dispatchGrid(uint32(colSize), limits); err != nil {
		return err
	}
	if _, _, err := dispatchGrid(uint32(ySize), limits); err != nil {
		return err
	}
	if _, _, err := groupGrid(uint64(p.Batch)*uint64(p.TilesPerMatrix), limits); err != nil {
		return err
	}
	col, err := b.take(uint64(colSize)*4, false)
	if err != nil {
		return err
	}
	defer b.put(col)
	im2col := visionParams{
		Kind: 0, N: uint32(colSize),
		D0: uint32(o.InChannels), D1: uint32(o.Height), D2: uint32(o.Width), D3: uint32(o.Groups),
		D4: uint32(o.KernelH), D5: uint32(o.KernelW), D6: uint32(oh), D7: uint32(ow),
		D8: uint32(o.StrideH), D9: uint32(o.StrideW), D10: uint32(o.PadH), D11: uint32(o.PadW),
		D12: uint32(o.DilationH), D13: uint32(o.DilationW),
	}
	biasOp := visionParams{
		Kind: 4, N: uint32(ySize),
		D0: uint32(o.OutChannels), D1: uint32(spatial), D2: 0, D3: 1,
	}
	return b.fail(b.driver.conv2d(conv2dBatch{
		out: y, input: x, weight: w, bias: v, col: col,
		outCount: uint32(ySize), inputCount: uint32(xSize), weightCount: uint32(wSize), biasCount: uint32(biasCount), colCount: uint32(colSize),
		im2col: im2col, biasOp: biasOp, gemm: p, variant: variant, hasBias: bias != nil,
	}))
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

func (b *Backend) MaxPool2d(out, x *gpu.Buffer, o gpu.Pool2dOptions) error {
	return b.pool2d(out, x, o, false)
}

func (b *Backend) AvgPool2d(out, x *gpu.Buffer, o gpu.Pool2dOptions) error {
	return b.pool2d(out, x, o, true)
}

func (b *Backend) pool2d(out, input *gpu.Buffer, o gpu.Pool2dOptions, avg bool) error {
	if o.Batch < 1 || o.Channels < 1 ||
		!validPoolAxis(o.Height, o.KernelH, o.StrideH, o.PadH, o.EndPadH, o.DilationH, o.OutH) ||
		!validPoolAxis(o.Width, o.KernelW, o.StrideW, o.PadW, o.EndPadW, o.DilationW, o.OutW) {
		return gpu.ErrInvalidArgument
	}
	if avg {
		if o.DilationH != 1 || o.DilationW != 1 {
			return gpu.ErrUnsupported
		}
		if o.IncludePad && (o.EndPadH > o.PadH || o.EndPadW > o.PadW) {
			return gpu.ErrUnsupported
		}
		if int64(o.Height)+2*int64(o.PadH) > math.MaxInt32 || int64(o.Width)+2*int64(o.PadW) > math.MaxInt32 {
			return gpu.ErrInvalidArgument
		}
	}
	xSize, err := gpu.Elements(o.Batch, o.Channels, o.Height, o.Width)
	if err != nil {
		return err
	}
	ySize, err := gpu.Elements(o.Batch, o.Channels, o.OutH, o.OutW)
	if err != nil {
		return err
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(input, xSize)
	if err != nil {
		return err
	}
	y, err := b.native(out, ySize)
	if err != nil {
		return err
	}
	if y == x {
		return fmt.Errorf("%w: pooling output aliases input", gpu.ErrInvalidArgument)
	}
	if _, _, err := dispatchGrid(uint32(ySize), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	kind := uint32(1)
	includePad := uint32(0)
	if avg {
		kind = 2
		if o.IncludePad {
			includePad = 1
		}
	}
	p := visionParams{
		Kind: kind, N: uint32(ySize),
		D0: uint32(o.Channels), D1: uint32(o.Height), D2: uint32(o.Width), D3: includePad,
		D4: uint32(o.KernelH), D5: uint32(o.KernelW), D6: uint32(o.OutH), D7: uint32(o.OutW),
		D8: uint32(o.StrideH), D9: uint32(o.StrideW), D10: uint32(o.PadH), D11: uint32(o.PadW),
		D12: uint32(o.DilationH), D13: uint32(o.DilationW),
	}
	return b.fail(b.driver.vision(y, x, x, x, uint32(ySize), uint32(xSize), uint32(xSize), uint32(xSize), p))
}

func (b *Backend) BatchNormEval(out, input, weight, bias, mean, variance *gpu.Buffer, outer, channels, inner int, eps float32) error {
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
	x, err := b.native(input, n)
	if err != nil {
		return err
	}
	m, err := b.native(mean, channels)
	if err != nil {
		return err
	}
	v, err := b.native(variance, channels)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	var w, bi *block
	if weight != nil {
		w, err = b.native(weight, channels)
		if err != nil {
			return err
		}
	}
	if bias != nil {
		bi, err = b.native(bias, channels)
		if err != nil {
			return err
		}
	}
	if y == x || y == m || y == v || (w != nil && y == w) || (bi != nil && y == bi) {
		return fmt.Errorf("%w: BatchNorm output aliases input", gpu.ErrInvalidArgument)
	}
	if _, _, err := dispatchGrid(uint32(n), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	norm := visionParams{Kind: 3, N: uint32(n), D0: uint32(channels), D1: uint32(inner), F0: eps}
	if err := b.fail(b.driver.vision(y, x, m, v, uint32(n), uint32(n), uint32(channels), uint32(channels), norm)); err != nil {
		return err
	}
	if w == nil && bi == nil {
		return nil
	}
	wBlock, wCount, hasW := y, uint32(n), uint32(0)
	if w != nil {
		wBlock, wCount, hasW = w, uint32(channels), 1
	}
	bBlock, bCount, hasB := y, uint32(n), uint32(0)
	if bi != nil {
		bBlock, bCount, hasB = bi, uint32(channels), 1
	}
	affine := visionParams{Kind: 4, N: uint32(n), D0: uint32(channels), D1: uint32(inner), D2: hasW, D3: hasB}
	return b.fail(b.driver.vision(y, y, wBlock, bBlock, uint32(n), uint32(n), wCount, bCount, affine))
}
