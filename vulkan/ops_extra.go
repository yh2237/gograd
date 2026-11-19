package vulkan

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/gpu"
)

var _ gpu.BinaryOpBackend = (*Backend)(nil)
var _ gpu.ScalarOpBackend = (*Backend)(nil)

func (b *Backend) BinaryOp(op gpu.BinaryOp, out, a, right *gpu.Buffer, n int) error {
	if op != gpu.BinarySub {
		return fmt.Errorf("%w: Vulkan binary op %d", gpu.ErrUnsupported, op)
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(a, n)
	if err != nil {
		return err
	}
	y, err := b.native(right, n)
	if err != nil {
		return err
	}
	z, err := b.native(out, n)
	if err != nil {
		return err
	}
	if z == x || z == y {
		return fmt.Errorf("%w: output aliases input", gpu.ErrInvalidArgument)
	}
	if n == 0 {
		return nil
	}
	if _, _, err := dispatchGrid(uint32(n), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	tmp, err := b.take(uint64(n)*4, false)
	if err != nil {
		return err
	}
	defer b.put(tmp)
	if err := b.fail(b.driver.pointwise(tmp, y, y, uint32(n), 2, -1)); err != nil {
		return err
	}
	return b.fail(b.driver.pointwise(z, x, tmp, uint32(n), 0, 0))
}

func (b *Backend) ScalarOp(op gpu.ScalarOp, out, a *gpu.Buffer, n int, value float32) error {
	switch op {
	case gpu.ScalarAdd, gpu.ScalarSub:
		if op == gpu.ScalarSub {
			value = -value
		}
		if err := b.lock(); err != nil {
			return err
		}
		defer b.mu.Unlock()
		x, err := b.native(a, n)
		if err != nil {
			return err
		}
		z, err := b.native(out, n)
		if err != nil {
			return err
		}
		if z == x {
			return fmt.Errorf("%w: output aliases input", gpu.ErrInvalidArgument)
		}
		if n == 0 {
			return nil
		}
		if _, _, err := dispatchGrid(uint32(n), b.driver.info().MaxWorkGroupCount); err != nil {
			return err
		}
		staging, err := b.take(4, true)
		if err != nil {
			return err
		}
		defer b.put(staging)
		scalarBuf, err := b.take(4, false)
		if err != nil {
			return err
		}
		defer b.put(scalarBuf)
		b.driver.write(staging, []float32{value})
		if err := b.fail(b.driver.copy(scalarBuf, staging, 4)); err != nil {
			return err
		}
		p := binaryParams{
			N:        uint32(n),
			Rank:     1,
			Op:       0,
			Shape:    [6]uint32{uint32(n)},
			StridesA: [6]uint32{1},
			StridesB: [6]uint32{0},
		}
		return b.fail(b.driver.binaryViews(z, x, scalarBuf, uint32(n), uint32(n), 1, p))
	case gpu.ScalarDiv:
		if value == 0 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Float32bits(value)&0x7fffff != 0 {
			return fmt.Errorf("%w: Vulkan scalar division by %v", gpu.ErrUnsupported, value)
		}
		inv := float32(1) / value
		mag := math.Abs(float64(inv))
		if !(mag >= 0x1p-126 && mag <= math.MaxFloat32) {
			return fmt.Errorf("%w: Vulkan scalar division reciprocal out of range", gpu.ErrUnsupported)
		}
		return b.elementwise(out, a, a, n, 2, inv)
	default:
		return fmt.Errorf("%w: Vulkan scalar op %d", gpu.ErrUnsupported, op)
	}
}
