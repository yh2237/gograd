package vulkan

import (
	"fmt"
	"slices"

	"github.com/yh2237/gograd/gpu"
)

// All push constants are uint32/float32, with the same layout as the GLSL
// blocks. Even GEMM's rank-four batch descriptor fits the core 128-byte limit.
type viewParams struct {
	N, Rank, Offset, Direction uint32
	Shape, Strides             [6]uint32
}

// binaryParams mirrors the GLSL block of binary.comp: the common shape and
// the two stride sets, plus the element offsets of the operands.
type binaryParams struct {
	N, Rank, Op, OffsetA, OffsetB uint32
	Shape                         [6]uint32
	StridesA                      [6]uint32
	StridesB                      [6]uint32
}

type gemmParams struct {
	M, N, K, Batch                           uint32
	AOffset, BOffset                         uint32
	ARow, ACol, BRow, BCol                   uint32
	BatchRank                                uint32
	BatchShape, ABatchStrides, BBatchStrides [4]uint32
	TileCols, TilesPerMatrix                 uint32
}

func (b *Backend) Gather(out *gpu.Buffer, src gpu.View) error  { return b.copyView(out, src, false) }
func (b *Backend) Scatter(dst gpu.View, src *gpu.Buffer) error { return b.copyView(src, dst, true) }

// AddViews evaluates a + b on two views with the same broadcast shape. The
// operands keep their strides, so no expansion buffer is needed.
func (b *Backend) AddViews(out *gpu.Buffer, a, right gpu.View) error {
	return b.binaryViews(out, a, right, 0)
}

// MulViews evaluates a * b on two views with the same broadcast shape.
func (b *Backend) MulViews(out *gpu.Buffer, a, right gpu.View) error {
	return b.binaryViews(out, a, right, 1)
}

func (b *Backend) binaryViews(out *gpu.Buffer, a, right gpu.View, kind uint32) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	n, err := a.Validate(b)
	if err != nil {
		return err
	}
	rn, err := right.Validate(b)
	if err != nil {
		return err
	}
	if n != rn || !slices.Equal(a.Shape, right.Shape) {
		return fmt.Errorf("%w: view operands must share a shape", gpu.ErrInvalidArgument)
	}
	if len(a.Shape) > 6 {
		return fmt.Errorf("%w: view rank", gpu.ErrUnsupported)
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	// The shader addresses the operands through their strides, so the
	// descriptor range has to cover every element the view can reach.
	usedA, usedB := viewSpan(a), viewSpan(right)
	x, err := b.native(a.Buffer, a.Buffer.Len())
	if err != nil {
		return err
	}
	w, err := b.native(right.Buffer, right.Buffer.Len())
	if err != nil {
		return err
	}
	if y == x || y == w {
		return fmt.Errorf("%w: view output aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	p := binaryParams{N: uint32(n), Rank: uint32(len(a.Shape)), Op: kind, OffsetA: uint32(a.Offset), OffsetB: uint32(right.Offset)}
	for i := range a.Shape {
		p.Shape[i] = uint32(a.Shape[i])
		p.StridesA[i] = uint32(a.Strides[i])
		p.StridesB[i] = uint32(right.Strides[i])
	}
	return b.fail(b.driver.binaryViews(y, x, w, uint32(n), uint32(usedA), uint32(usedB), p))
}

// viewSpan reports the number of elements a view can reach from its offset.
func viewSpan(v gpu.View) int {
	span := 1
	for i, stride := range v.Strides {
		if v.Shape[i] > 1 {
			span += (v.Shape[i] - 1) * stride
		}
	}
	return v.Offset + span
}

func (b *Backend) copyView(linear *gpu.Buffer, view gpu.View, scatter bool) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	n, err := view.Validate(b)
	if scatter {
		n, err = view.ValidateWrite(b)
	}
	if err != nil {
		return err
	}
	base, err := b.native(view.Buffer, view.Buffer.Len())
	if err != nil {
		return err
	}
	flat, err := b.native(linear, n)
	if err != nil {
		return err
	}
	if base == flat {
		return fmt.Errorf("%w: view copy aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	if _, _, err := dispatchGrid(uint32(n), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	p := viewParams{N: uint32(n), Rank: uint32(len(view.Shape)), Offset: uint32(view.Offset)}
	for i := range view.Shape {
		p.Shape[i], p.Strides[i] = uint32(view.Shape[i]), uint32(view.Strides[i])
	}
	if scatter {
		p.Direction = 1
		return b.fail(b.driver.viewCopy(base, flat, uint32(view.Buffer.Len()), uint32(n), p))
	}
	return b.fail(b.driver.viewCopy(flat, base, uint32(n), uint32(view.Buffer.Len()), p))
}

func (b *Backend) MatMul(out, a, right *gpu.Buffer, m, n, k int, transA, transB bool) error {
	mn, err := gpu.Elements(m, n)
	if err != nil {
		return err
	}
	mk, err := gpu.Elements(m, k)
	if err != nil {
		return err
	}
	kn, err := gpu.Elements(k, n)
	if err != nil {
		return err
	}
	for _, input := range []struct {
		buffer *gpu.Buffer
		n      int
	}{{out, mn}, {a, mk}, {right, kn}} {
		if err := gpu.CheckBuffer(b, input.buffer, input.n); err != nil {
			return err
		}
	}
	as, bs := []int{k, 1}, []int{n, 1}
	if transA {
		as = []int{1, m}
	}
	if transB {
		bs = []int{1, k}
	}
	return b.MatMulViews(out, gpu.View{Buffer: a, Shape: []int{m, k}, Strides: as}, gpu.View{Buffer: right, Shape: []int{k, n}, Strides: bs})
}

func (b *Backend) MatMulViews(out *gpu.Buffer, a, right gpu.View) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	if _, err := a.Validate(b); err != nil {
		return err
	}
	if _, err := right.Validate(b); err != nil {
		return err
	}
	shape, err := gpu.MatMulShape(a.Shape, right.Shape)
	if err != nil {
		return err
	}
	n, err := gpu.Elements(shape...)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	x, err := b.native(a.Buffer, a.Buffer.Len())
	if err != nil {
		return err
	}
	w, err := b.native(right.Buffer, right.Buffer.Len())
	if err != nil {
		return err
	}
	if y == x || y == w {
		return fmt.Errorf("%w: GEMM output aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	k := a.Shape[len(a.Shape)-1]
	if k == 0 {
		return b.fail(b.driver.zero(y, uint64(n)*4))
	}
	p, variant := matrixParams(a, right, shape)
	if _, _, err := groupGrid(uint64(p.Batch)*uint64(p.TilesPerMatrix), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	return b.fail(b.driver.gemm(y, x, w, uint32(n), uint32(a.Buffer.Len()), uint32(right.Buffer.Len()), p, variant))
}

func matrixParams(a, b gpu.View, shape []int) (gemmParams, int) {
	rank := len(shape) - 2
	p := gemmParams{M: uint32(shape[len(shape)-2]), N: uint32(shape[len(shape)-1]), K: uint32(a.Shape[len(a.Shape)-1]), Batch: 1, AOffset: uint32(a.Offset), BOffset: uint32(b.Offset), BatchRank: uint32(rank)}
	for i, dim := range shape[:rank] {
		p.Batch *= uint32(dim)
		p.BatchShape[i] = uint32(dim)
	}
	fill := func(v gpu.View, dst *[4]uint32) {
		srcRank := len(v.Shape) - 2
		for i := 0; i < rank; i++ {
			if j := i - rank + srcRank; j >= 0 && v.Shape[j] != 1 {
				dst[i] = uint32(v.Strides[j])
			}
		}
	}
	fill(a, &p.ABatchStrides)
	fill(b, &p.BBatchStrides)
	p.ARow, p.ACol = uint32(a.Strides[len(a.Strides)-2]), uint32(a.Strides[len(a.Strides)-1])
	p.BRow, p.BCol = uint32(b.Strides[len(b.Strides)-2]), uint32(b.Strides[len(b.Strides)-1])
	variant := 0
	if p.ARow < p.ACol {
		p.ARow, p.ACol = p.ACol, p.ARow
		variant |= 2
	}
	if p.BRow < p.BCol {
		p.BRow, p.BCol = p.BCol, p.BRow
		variant |= 1
	}
	p.TileCols = (p.N + 15) / 16
	p.TilesPerMatrix = ((p.M + 7) / 8) * p.TileCols
	return p, variant
}
