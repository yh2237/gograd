package directml

import (
	"github.com/yh2237/gograd/gpu"
	"slices"
)

var _ gpu.ViewBackend = (*Backend)(nil)
var _ gpu.ViewMatMuler = (*Backend)(nil)
var _ gpu.ViewBinaryBackend = (*Backend)(nil)

func (b *Backend) Gather(out *gpu.Buffer, src gpu.View) error  { return b.copyView(out, src, false) }
func (b *Backend) Scatter(dst gpu.View, src *gpu.Buffer) error { return b.copyView(src, dst, true) }
func (b *Backend) copyView(linear *gpu.Buffer, v gpu.View, scatter bool) error {
	if err := b.lock(); err != nil {
		return err
	}
	defer b.mu.Unlock()
	n, err := v.Validate(b)
	if scatter {
		n, err = v.ValidateWrite(b)
	}
	if err != nil {
		return err
	}
	base, err := b.native(v.Buffer, v.Buffer.Len())
	if err != nil {
		return err
	}
	flat, err := b.native(linear, n)
	if err != nil {
		return err
	}
	if base == flat {
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	if scatter {
		return b.fail(b.driver.copyView(base, flat, v, true))
	}
	return b.gatherLocked(flat, base, v)
}

func (b *Backend) gatherLocked(out, src *allocation, v gpu.View) error {
	if v.Offset%4 == 0 && len(v.Shape) <= 4 && !v.Compact() {
		a := viewLayout(v)
		s := a.sizes
		y := compactLayout(s[0], s[1], s[2], s[3])
		n, _ := gpu.Elements(v.Shape...)
		return b.fail(b.driver.executeAt(opKey{kind: opIdentity, n: n, aLayout: a, yLayout: y}, out, resourceBinding{src, uint64(v.Offset) * 4}))
	}
	return b.fail(b.driver.copyView(out, src, v, false))
}

// materializeLocked copies only on the GPU. In particular, byte offsets need
// not satisfy DirectML's tensor binding alignment: D3D12 buffer copies handle
// those offsets before the aligned, contiguous result is bound to an operator.
func (b *Backend) materializeLocked(v gpu.View) (*allocation, func(), error) {
	n, err := v.Validate(b)
	if err != nil {
		return nil, nil, err
	}
	base, err := b.native(v.Buffer, v.Buffer.Len())
	if err != nil {
		return nil, nil, err
	}
	if v.Offset == 0 && v.Compact() && v.Buffer.Len() == n {
		return base, func() {}, nil
	}
	out, err := b.take(uint64(n)*4, heapDefault)
	if err != nil {
		return nil, nil, err
	}
	if n != 0 {
		if err := b.gatherLocked(out, base, v); err != nil {
			b.put(out)
			return nil, nil, err
		}
	}
	return out, func() { b.put(out) }, nil
}

func (b *Backend) MatMulViews(out *gpu.Buffer, a, right gpu.View) error {
	if err := b.lock(); err != nil {
		return err
	}
	if _, err := a.Validate(b); err != nil {
		b.mu.Unlock()
		return err
	}
	if _, err := right.Validate(b); err != nil {
		b.mu.Unlock()
		return err
	}
	shape, err := gpu.MatMulShape(a.Shape, right.Shape)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	n, err := gpu.Elements(shape...)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	y, err := b.native(out, n)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	x, err := b.native(a.Buffer, a.Buffer.Len())
	if err != nil {
		b.mu.Unlock()
		return err
	}
	w, err := b.native(right.Buffer, right.Buffer.Len())
	if err != nil {
		b.mu.Unlock()
		return err
	}
	if y == x || y == w {
		b.mu.Unlock()
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		b.mu.Unlock()
		return nil
	}
	m, cols, k := shape[len(shape)-2], shape[len(shape)-1], a.Shape[len(a.Shape)-1]
	if k == 0 {
		b.mu.Unlock()
		return b.Zero(out)
	}
	defer b.mu.Unlock()
	batchShape := shape[:len(shape)-2]
	as := append(append([]int(nil), batchShape...), m, k)
	bs := append(append([]int(nil), batchShape...), k, cols)
	av, err := a.BroadcastTo(as)
	if err != nil {
		return err
	}
	bv, err := right.BroadcastTo(bs)
	if err != nil {
		return err
	}
	// Ordinary batch/head transposes and expansions can be read directly by
	// baseline four-axis GEMM. No expanded matrices or transpose copies needed.
	if a.Offset%4 == 0 && right.Offset%4 == 0 && len(shape) <= 4 {
		avLayout, bvLayout := viewLayout(av), viewLayout(bv)
		ys := [4]uint32{1, 1, 1, 1}
		for i, d := range shape {
			ys[4-len(shape)+i] = uint32(d)
		}
		yLayout := compactLayout(ys[0], ys[1], ys[2], ys[3])
		return b.fail(b.driver.executeAt(opKey{kind: opGEMM, m: m, n: cols, k: k, aLayout: avLayout, bLayout: bvLayout, yLayout: yLayout}, y, resourceBinding{x, uint64(a.Offset) * 4}, resourceBinding{w, uint64(right.Offset) * 4}))
	}
	x, releaseA, err := b.materializeLocked(av)
	if err != nil {
		return err
	}
	defer releaseA()
	w, releaseB, err := b.materializeLocked(bv)
	if err != nil {
		return err
	}
	defer releaseB()
	batch, err := gpu.Elements(batchShape...)
	if err != nil {
		return err
	}
	if batch == 1 {
		batch = 0
	} // Reuse the M4 cache entries for ordinary matrices.
	return b.fail(b.driver.execute(opKey{kind: opGEMM, m: m, n: cols, k: k, batch: batch}, y, x, w))
}

func viewLayout(v gpu.View) tensorLayout {
	layout := tensorLayout{sizes: [4]uint32{1, 1, 1, 1}}
	for i, dim := range v.Shape {
		axis := 4 - len(v.Shape) + i
		layout.sizes[axis] = uint32(dim)
		layout.strides[axis] = uint32(v.Strides[i])
	}
	return layout
}

func (b *Backend) AddViews(out *gpu.Buffer, a, right gpu.View) error {
	return b.binaryViews(out, a, right, opAdd)
}
func (b *Backend) MulViews(out *gpu.Buffer, a, right gpu.View) error {
	return b.binaryViews(out, a, right, opMul)
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
		return gpu.ErrInvalidArgument
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
		return gpu.ErrInvalidArgument
	}
	if n == 0 {
		return nil
	}
	if len(a.Shape) > 4 || a.Offset%4 != 0 || right.Offset%4 != 0 {
		return gpu.ErrUnsupported
	}
	al, bl := viewLayout(a), viewLayout(right)
	s := al.sizes
	yl := compactLayout(s[0], s[1], s[2], s[3])
	return b.fail(b.driver.executeAt(opKey{kind: kind, n: n, aLayout: al, bLayout: bl, yLayout: yl}, y, resourceBinding{x, uint64(a.Offset) * 4}, resourceBinding{w, uint64(right.Offset) * 4}))
}
