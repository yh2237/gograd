package directml

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/gpu"
)

var (
	_ gpu.ExactGELUBackend = (*Backend)(nil)
	_ gpu.FusedBiasBackend = (*Backend)(nil)
)

func compactLayout(a, b, c, d uint32) tensorLayout {
	return tensorLayout{sizes: [4]uint32{a, b, c, d}, strides: [4]uint32{b * c * d, c * d, d, 1}}
}

// Interpret time-major data as the NCHW tensor required by DML convolution.
func ntcLayout(batch, frames, channels int) tensorLayout {
	return tensorLayout{sizes: [4]uint32{uint32(batch), uint32(channels), 1, uint32(frames)}, strides: [4]uint32{uint32(frames * channels), 1, uint32(frames * channels), uint32(channels)}}
}

func (b *Backend) unary(out, a *gpu.Buffer, n int, key opKey) error {
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
	if y == x {
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	if key.kind == opGELUExact && b.driver.info().FeatureLevel < 0x5100 {
		return fmt.Errorf("%w: native GELU requires DirectML feature level 5.1", gpu.ErrUnsupported)
	}
	if key.kind == opErf && b.driver.info().FeatureLevel < 0x2000 {
		return gpu.ErrUnsupported
	}
	return b.fail(b.driver.execute(key, y, x))
}
func (b *Backend) GELUExact(out, a *gpu.Buffer, n int) error {
	return b.unary(out, a, n, opKey{kind: opGELUExact, n: n})
}

func (b *Backend) GELU(out, a *gpu.Buffer, n int) error {
	return b.unary(out, a, n, opKey{kind: opGELUTanh, n: n})
}

func (b *Backend) Softmax(out, a *gpu.Buffer, rows, cols int) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 {
		return gpu.ErrInvalidArgument
	}
	layout := compactLayout(1, 1, uint32(rows), uint32(cols))
	return b.unary(out, a, n, opKey{kind: opSoftmax, n: n, aLayout: layout, yLayout: layout})
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
		return gpu.ErrInvalidArgument
	}
	layout := compactLayout(uint32(rows), 1, 1, uint32(cols))
	affine := tensorLayout{sizes: [4]uint32{uint32(rows), 1, 1, uint32(cols)}, strides: [4]uint32{0, 0, 0, 1}}
	return b.normalization(out, a, weight, bias, n, cols, layout, affine, cols, eps)
}
func (b *Backend) GroupNorm(out, a, weight, bias *gpu.Buffer, batch, frames, channels, groups int, eps float32) error {
	n, err := gpu.Elements(batch, frames, channels)
	if err != nil {
		return err
	}
	if frames == 0 || channels == 0 || groups < 1 || channels%groups != 0 || !validEps(eps) {
		return gpu.ErrInvalidArgument
	}
	cg := uint32(channels / groups)
	layout := tensorLayout{sizes: [4]uint32{uint32(batch), uint32(groups), uint32(frames), cg}, strides: [4]uint32{uint32(frames * channels), cg, uint32(channels), 1}}
	affine := tensorLayout{sizes: [4]uint32{uint32(batch), uint32(groups), uint32(frames), cg}, strides: [4]uint32{0, cg, 0, 1}}
	return b.normalization(out, a, weight, bias, n, channels, layout, affine, frames*(channels/groups), eps)
}

func (b *Backend) normalization(out, a, weight, bias *gpu.Buffer, n, channels int, layout, affine tensorLayout, reduced int, eps float32) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	x, err := b.native(a, n)
	if err != nil {
		return err
	}
	w, err := b.native(weight, channels)
	if err != nil {
		return err
	}
	v, err := b.native(bias, channels)
	if err != nil {
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		return err
	}
	if y == x || y == w || y == v {
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	if reduced == 1 {
		// Some DirectML implementations report NaN variance for a 1-element group.
		flat := compactLayout(1, 1, uint32(n/channels), uint32(channels))
		flatAffine := flat
		flatAffine.strides = [4]uint32{0, 0, 0, 1}
		key := opKey{kind: opIdentity, n: n, aLayout: flatAffine, yLayout: flat}
		return b.fail(b.driver.executeAt(key, y, resourceBinding{v, 0}))
	}
	key := opKey{kind: opNorm, n: n, scalar: math.Float32bits(eps), aLayout: layout, bLayout: affine, yLayout: layout}
	return b.fail(b.driver.execute(key, y, x, w, v))
}

func (b *Backend) BiasGELU(out, a, bias *gpu.Buffer, rows, cols int) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols < 1 {
		return gpu.ErrInvalidArgument
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, n, bufferArg{a, n}, bufferArg{bias, cols})
	if err != nil || n == 0 {
		return err
	}
	return b.fail(b.driver.execute(opKey{kind: opBiasGELU, n: n, scalar: uint32(cols)}, y, inputs...))
}

func (b *Backend) BiasResidual(out, a, bias, residual *gpu.Buffer, rows, cols int) error {
	n, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols < 1 {
		return gpu.ErrInvalidArgument
	}
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	y, inputs, err := b.opBuffers(out, n, bufferArg{a, n}, bufferArg{bias, cols}, bufferArg{residual, n})
	if err != nil || n == 0 {
		return err
	}
	return b.fail(b.driver.execute(opKey{kind: opBiasResidual, n: n, scalar: uint32(cols)}, y, inputs...))
}

func (b *Backend) Conv1d(out, input, weight, bias *gpu.Buffer, o gpu.Conv1dOptions) error {
	length, err := o.OutputLength()
	if err != nil {
		return err
	}
	// DirectML writes zeros instead of the convolution when a dilated kernel
	// spans the padded input exactly and the input length equals the kernel,
	// which leaves a single output position. Reject it rather than return
	// silently wrong values.
	if length == 1 && o.Dilation > 1 && o.InputLength == o.Kernel {
		return fmt.Errorf("%w: DirectML Conv1d single-position dilated window", gpu.ErrUnsupported)
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
	var v *allocation
	if bias != nil {
		v, err = b.native(bias, o.OutChannels)
		if err != nil {
			return err
		}
	}
	if y == x || y == w || y == v {
		return gpu.ErrInvalidArgument
	}
	return b.fail(b.driver.execute(opKey{kind: opConv, n: length, conv: o, hasBias: bias != nil}, y, x, w, v))
}
