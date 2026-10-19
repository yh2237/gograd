package vulkan

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/gpu"
)

type reduceParams struct {
	Rows, Cols, Kind, Frames, Channels, Groups uint32
	Eps                                        float32
}

type convParams struct {
	N, InputLength, InChannels, Kernel, OutputLength uint32
	Stride, Padding, Dilation, Kind, OutChannels     uint32
}

// convBatch is a fixed, internal 2/3-dispatch batch. Its scratch and input blocks
// stay owned by the caller until the single synchronous submission returns.
type convBatch struct {
	out, input, weight, bias, col                          *block
	outCount, inputCount, weightCount, biasCount, colCount uint32
	conv                                                   convParams
	gemm                                                   gemmParams
	variant                                                int
	hasBias                                                bool
}

var _ gpu.ExactGELUBackend = (*Backend)(nil)

func (b *Backend) GELU(out, a *gpu.Buffer, n int) error {
	return b.elementwise(out, a, a, n, 3, 0)
}

func (b *Backend) GELUExact(out, a *gpu.Buffer, n int) error {
	return b.elementwise(out, a, a, n, 4, 0)
}

func (b *Backend) Softmax(out, a *gpu.Buffer, rows, cols int) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 {
		return fmt.Errorf("%w: empty softmax axis", gpu.ErrInvalidArgument)
	}
	return b.reduction(out, a, nil, nil, n, 0, reduceParams{Rows: uint32(rows), Cols: uint32(cols)})
}

func validEps(eps float32) bool {
	return eps > 0 && !math.IsInf(float64(eps), 0) && !math.IsNaN(float64(eps))
}

func (b *Backend) LayerNorm(out, a, weight, bias *gpu.Buffer, rows, cols int, eps float32) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 || !validEps(eps) {
		return fmt.Errorf("%w: LayerNorm dimensions/epsilon", gpu.ErrInvalidArgument)
	}
	return b.reduction(out, a, weight, bias, n, cols, reduceParams{Rows: uint32(rows), Cols: uint32(cols), Kind: 1, Eps: eps})
}

func (b *Backend) GroupNorm(out, a, weight, bias *gpu.Buffer, batch, frames, channels, groups int, eps float32) error {
	n, err := gpu.Elements(batch, frames, channels)
	if err != nil {
		return err
	}
	if frames == 0 || channels == 0 || groups < 1 || channels%groups != 0 || !validEps(eps) {
		return fmt.Errorf("%w: GroupNorm dimensions/epsilon", gpu.ErrInvalidArgument)
	}
	rows, err := gpu.Elements(batch, groups)
	if err != nil {
		return err
	}
	cols, err := gpu.Elements(frames, channels/groups)
	if err != nil {
		return err
	}
	return b.reduction(out, a, weight, bias, n, channels, reduceParams{Rows: uint32(rows), Cols: uint32(cols), Kind: 2, Frames: uint32(frames), Channels: uint32(channels), Groups: uint32(groups), Eps: eps})
}

func (b *Backend) reduction(out, a, weight, bias *gpu.Buffer, n, affineCount int, p reduceParams) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(a, n)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	w, v := x, x
	if p.Kind != 0 {
		w, err = b.native(weight, affineCount)
		if err != nil {
			return err
		}
		v, err = b.native(bias, affineCount)
		if err != nil {
			return err
		}
	} else {
		affineCount = n
	}
	if y == x || y == w || y == v {
		return fmt.Errorf("%w: reduction output aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	if _, _, err := groupGrid(uint64(p.Rows), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	return b.fail(b.driver.reduce(y, x, w, v, uint32(n), uint32(affineCount), p))
}

// Conv1d lowers [batch,time,channels] into [batch*outTime,inChannels*kernel]
// on the device, multiplies by the transposed weights and adds channel bias.
// One bounded-pool scratch allocation holds im2col and is returned on failure.
func (b *Backend) Conv1d(out, input, weight, bias *gpu.Buffer, o gpu.Conv1dOptions) error {
	length, err := o.OutputLength()
	if err != nil {
		return err
	}
	xSize, err := gpu.Elements(o.Batch, o.InputLength, o.InChannels)
	if err != nil {
		return err
	}
	wSize, err := gpu.Elements(o.OutChannels, o.InChannels, o.Kernel)
	if err != nil {
		return err
	}
	ySize, err := gpu.Elements(o.Batch, length, o.OutChannels)
	if err != nil {
		return err
	}
	colSize, err := gpu.Elements(o.Batch, length, o.InChannels, o.Kernel)
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
	v := x // A valid unused descriptor when bias is absent.
	biasCount := xSize
	if bias != nil {
		v, err = b.native(bias, o.OutChannels)
		if err != nil {
			return err
		}
		biasCount = o.OutChannels
	}
	if y == x || y == w || y == v {
		return fmt.Errorf("%w: Conv1d output aliases input", gpu.ErrInvalidArgument)
	}
	if uint64(colSize)*4 > b.driver.info().MaxStorageBufferBytes {
		return fmt.Errorf("%w: Conv1d im2col exceeds storage buffer range", gpu.ErrUnsupported)
	}
	depth := o.InChannels * o.Kernel
	p, variant := matrixParams(
		gpu.View{Shape: []int{o.Batch * length, depth}, Strides: []int{depth, 1}},
		gpu.View{Shape: []int{depth, o.OutChannels}, Strides: []int{1, depth}},
		[]int{o.Batch * length, o.OutChannels})
	if _, _, err := dispatchGrid(uint32(colSize), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	if _, _, err := dispatchGrid(uint32(ySize), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	if _, _, err := groupGrid(uint64(p.TilesPerMatrix), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	col, err := b.take(uint64(colSize)*4, false)
	if err != nil {
		return err
	}
	defer b.put(col)
	cp := convParams{N: uint32(colSize), InputLength: uint32(o.InputLength), InChannels: uint32(o.InChannels), Kernel: uint32(o.Kernel), OutputLength: uint32(length), Stride: uint32(o.Stride), Padding: uint32(o.Padding), Dilation: uint32(o.Dilation), OutChannels: uint32(o.OutChannels)}
	return b.fail(b.driver.conv1d(convBatch{
		out: y, input: x, weight: w, bias: v, col: col,
		outCount: uint32(ySize), inputCount: uint32(xSize), weightCount: uint32(wSize), biasCount: uint32(biasCount), colCount: uint32(colSize),
		conv: cp, gemm: p, variant: variant, hasBias: bias != nil,
	}))
}
