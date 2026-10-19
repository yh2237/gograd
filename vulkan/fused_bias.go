package vulkan

import "github.com/yh2237/gograd/gpu"

var _ gpu.FusedBiasBackend = (*Backend)(nil)

const (
	pointwiseBiasGELU     uint32 = 5
	pointwiseBiasResidual uint32 = 6
)

func (b *Backend) BiasGELU(out, a, bias *gpu.Buffer, rows, cols int) error {
	return b.fusedBias(out, a, bias, nil, rows, cols, pointwiseBiasGELU)
}
func (b *Backend) BiasResidual(out, a, bias, residual *gpu.Buffer, rows, cols int) error {
	return b.fusedBias(out, a, bias, residual, rows, cols, pointwiseBiasResidual)
}
func (b *Backend) fusedBias(out, a, bias, residual *gpu.Buffer, rows, cols int, kind uint32) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 {
		return gpu.ErrInvalidArgument
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(a, n)
	if err != nil {
		return err
	}
	w, err := b.native(bias, cols)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	r := x // Valid unused descriptor for BiasGELU.
	if kind == pointwiseBiasResidual {
		r, err = b.native(residual, n)
		if err != nil {
			return err
		}
	}
	if y == x || y == w || y == r {
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	if _, _, err := dispatchGrid(uint32(n), b.driver.info().MaxWorkGroupCount); err != nil {
		return err
	}
	return b.fail(b.driver.fusedBias(y, x, w, r, uint32(n), uint32(cols), kind))
}
