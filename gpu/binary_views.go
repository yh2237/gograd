package gpu

// ViewBinaryBackend optionally evaluates elementwise views without expansion
// buffers. Inputs already have the same broadcast shape. Implementations may
// return ErrUnsupported for a layout, permitting device-only materialization by
// the caller; other errors must be propagated, not silently retried on the CPU.
type ViewBinaryBackend interface {
	AddViews(out *Buffer, a, b View) error
	MulViews(out *Buffer, a, b View) error
}
