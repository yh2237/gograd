package gpu

import "fmt"

// View describes logical float32 elements in a buffer. Strides and Offset are
// element indices, not bytes. Views borrow the buffer and do not own a lease.
// Rank is limited to six; zero strides implement expansion/broadcasting.
type View struct {
	Buffer         *Buffer
	Shape, Strides []int
	Offset         int
}

// ViewBackend optionally materializes views or writes into a non-overlapping
// view, entirely on the device. Scatter replaces values; it is not a gradient
// accumulation primitive. Source/output buffers must not alias the view buffer.
type ViewBackend interface {
	Gather(out *Buffer, src View) error
	Scatter(dst View, src *Buffer) error
}

// ViewMatMuler implements [...,m,k] @ [...,k,n], broadcasting batch axes.
// Inputs may be non-contiguous, sliced, transposed or expanded views.
type ViewMatMuler interface {
	MatMulViews(out *Buffer, a, b View) error
}

func (v View) Validate(owner Backend) (int, error) {
	if len(v.Shape) > 6 || len(v.Shape) != len(v.Strides) {
		return 0, fmt.Errorf("%w: GPU view rank/strides", ErrInvalidArgument)
	}
	n, err := Elements(v.Shape...)
	if err != nil {
		return 0, err
	}
	if _, err := v.Buffer.Native(owner); err != nil {
		return 0, err
	}
	if v.Offset < 0 || v.Offset > v.Buffer.Len() {
		return 0, fmt.Errorf("%w: GPU view offset", ErrInvalidArgument)
	}
	last := uint64(v.Offset)
	for i, stride := range v.Strides {
		if stride < 0 || stride > maxElements {
			return 0, fmt.Errorf("%w: GPU view stride", ErrInvalidArgument)
		}
		if n != 0 {
			last += uint64(v.Shape[i]-1) * uint64(stride)
		}
	}
	if n != 0 && last >= uint64(v.Buffer.Len()) {
		return 0, fmt.Errorf("%w: GPU view exceeds allocation", ErrInvalidArgument)
	}
	return n, nil
}

// ValidateWrite rejects overlap conservatively using sorted strides. This
// covers all non-overlapping permutations/slices of contiguous gograd storage;
// expanded axes with more than one element are never writable.
func (v View) ValidateWrite(owner Backend) (int, error) {
	n, err := v.Validate(owner)
	if err != nil || n == 0 {
		return n, err
	}
	var axes [6]struct{ stride, size uint64 }
	count := 0
	for i, dim := range v.Shape {
		if dim <= 1 {
			continue
		}
		axis := struct{ stride, size uint64 }{uint64(v.Strides[i]), uint64(dim)}
		j := count
		for j > 0 && axes[j-1].stride > axis.stride {
			axes[j] = axes[j-1]
			j--
		}
		axes[j] = axis
		count++
	}
	span := uint64(1)
	for _, axis := range axes[:count] {
		if axis.stride < span {
			return 0, fmt.Errorf("%w: overlapping GPU view write", ErrInvalidArgument)
		}
		span += (axis.size - 1) * axis.stride
	}
	return n, nil
}

// Compact reports a row-major logical layout, ignoring strides on singleton
// axes. Offset and backing buffer size do not affect layout compactness.
func (v View) Compact() bool {
	if len(v.Shape) > 6 || len(v.Shape) != len(v.Strides) {
		return false
	}
	n, err := Elements(v.Shape...)
	if err != nil {
		return false
	}
	for _, stride := range v.Strides {
		if stride < 0 {
			return false
		}
	}
	if n == 0 {
		return true
	}
	expected := 1
	for i := len(v.Shape) - 1; i >= 0; i-- {
		if v.Shape[i] > 1 && v.Strides[i] != expected {
			return false
		}
		expected *= v.Shape[i]
	}
	return true
}

func BroadcastShape(a, b []int) ([]int, error) {
	rank := max(len(a), len(b))
	if rank > 6 {
		return nil, fmt.Errorf("%w: broadcast rank", ErrInvalidArgument)
	}
	out := make([]int, rank)
	for i := 0; i < rank; i++ {
		x, y := 1, 1
		if j := i - rank + len(a); j >= 0 {
			x = a[j]
		}
		if j := i - rank + len(b); j >= 0 {
			y = b[j]
		}
		switch {
		case x == y:
			out[i] = x
		case x == 1:
			out[i] = y
		case y == 1:
			out[i] = x
		default:
			return nil, fmt.Errorf("%w: broadcast shapes %v and %v", ErrInvalidArgument, a, b)
		}
	}
	if _, err := Elements(out...); err != nil {
		return nil, err
	}
	return out, nil
}

func MatMulShape(a, b []int) ([]int, error) {
	if len(a) < 2 || len(b) < 2 || len(a) > 6 || len(b) > 6 || a[len(a)-1] != b[len(b)-2] {
		return nil, fmt.Errorf("%w: matmul dimensions", ErrInvalidArgument)
	}
	if _, err := Elements(a...); err != nil {
		return nil, err
	}
	if _, err := Elements(b...); err != nil {
		return nil, err
	}
	batch, err := BroadcastShape(a[:len(a)-2], b[:len(b)-2])
	if err != nil {
		return nil, err
	}
	shape := append(batch, a[len(a)-2], b[len(b)-1])
	if _, err := Elements(shape...); err != nil {
		return nil, err
	}
	return shape, nil
}

// BroadcastTo returns metadata only; it neither allocates storage nor downloads
// values. The returned view shares the original buffer.
func (v View) BroadcastTo(shape []int) (View, error) {
	if len(shape) < len(v.Shape) || len(shape) > 6 || len(v.Shape) != len(v.Strides) {
		return View{}, fmt.Errorf("%w: broadcast view rank", ErrInvalidArgument)
	}
	if _, err := Elements(shape...); err != nil {
		return View{}, err
	}
	strides := make([]int, len(shape))
	pad := len(shape) - len(v.Shape)
	for i, d := range shape {
		if i < pad {
			continue
		}
		old := v.Shape[i-pad]
		if old == d {
			strides[i] = v.Strides[i-pad]
		} else if old != 1 {
			return View{}, fmt.Errorf("%w: broadcast view shape", ErrInvalidArgument)
		}
	}
	return View{Buffer: v.Buffer, Shape: append([]int(nil), shape...), Strides: strides, Offset: v.Offset}, nil
}
