package directml

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/gpu"
)

var _ gpu.ExactGELUBackend = (*Backend)(nil)

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
	return b.fail(b.driver.execute(key, y, x))
}
func (b *Backend) GELUExact(out, a *gpu.Buffer, n int) error {
	return b.unary(out, a, n, opKey{kind: opGELUExact, n: n})
}

// GELU preserves the backend's tanh-approximation contract. DML_ACTIVATION_GELU
// is the erf-based variant, so build this variant from GPU arithmetic and tanh.
func (b *Backend) GELU(out, a *gpu.Buffer, n int) error {
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
	u, err := b.take(uint64(n)*4, heapDefault)
	if err != nil {
		return err
	}
	defer b.put(u)
	v, err := b.take(uint64(n)*4, heapDefault)
	if err != nil {
		return err
	}
	defer b.put(v)
	scalar := func(out, in *allocation, scale, bias float32) error {
		return b.driver.execute(opKey{kind: opScalar, n: n, scalar: math.Float32bits(scale), bias: math.Float32bits(bias)}, out, in)
	}
	steps := []func() error{
		func() error { return b.driver.execute(opKey{kind: opMul, n: n}, u, x, x) },
		func() error { return b.driver.execute(opKey{kind: opMul, n: n}, v, u, x) },
		func() error { return scalar(u, v, .044715, 0) },
		func() error { return b.driver.execute(opKey{kind: opAdd, n: n}, v, x, u) },
		func() error { return scalar(u, v, float32(math.Sqrt(2/math.Pi)), 0) },
		func() error { return b.driver.execute(opKey{kind: opTanh, n: n}, v, u) },
		func() error { return scalar(u, v, 1, 1) },
		func() error { return b.driver.execute(opKey{kind: opMul, n: n}, v, x, u) },
		func() error { return scalar(y, v, .5, 0) },
	}
	for _, step := range steps {
		if err := b.fail(step()); err != nil {
			return err
		}
	}
	return nil
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
	return b.normalization(out, a, weight, bias, n, cols, compactLayout(uint32(rows), 1, 1, uint32(cols)), eps)
}
func (b *Backend) GroupNorm(out, a, weight, bias *gpu.Buffer, batch, frames, channels, groups int, eps float32) error {
	n, err := gpu.Elements(batch, frames, channels)
	if err != nil {
		return err
	}
	if frames == 0 || channels == 0 || groups < 1 || channels%groups != 0 || !validEps(eps) {
		return gpu.ErrInvalidArgument
	}
	layout := tensorLayout{sizes: [4]uint32{uint32(batch), uint32(groups), uint32(frames), uint32(channels / groups)}, strides: [4]uint32{uint32(frames * channels), uint32(channels / groups), uint32(channels), 1}}
	return b.normalization(out, a, weight, bias, n, channels, layout, eps)
}

func (b *Backend) normalization(out, a, weight, bias *gpu.Buffer, n, channels int, layout tensorLayout, eps float32) error {
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
	norm, err := b.take(uint64(n)*4, heapDefault)
	if err != nil {
		return err
	}
	defer b.put(norm)
	scaled, err := b.take(uint64(n)*4, heapDefault)
	if err != nil {
		return err
	}
	defer b.put(scaled)
	if err := b.fail(b.driver.execute(opKey{kind: opNorm, n: n, scalar: math.Float32bits(eps), aLayout: layout, yLayout: layout}, norm, x)); err != nil {
		return err
	}
	flat := compactLayout(1, 1, uint32(n/channels), uint32(channels))
	affine := flat
	affine.strides = [4]uint32{0, 0, 0, 1}
	key := opKey{kind: opMul, n: n, aLayout: flat, bLayout: affine, yLayout: flat}
	if err := b.fail(b.driver.execute(key, scaled, norm, w)); err != nil {
		return err
	}
	key.kind = opAdd
	return b.fail(b.driver.execute(key, y, scaled, v))
}

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
