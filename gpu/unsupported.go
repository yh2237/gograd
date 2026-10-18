package gpu

import "fmt"

// Copier is an optional device-to-device copy capability. Implementations must
// copy the complete logical buffers without transferring data through the CPU.
type Copier interface {
	Copy(out, src *Buffer, n int) error
}

// UnsupportedOps can be embedded by a backend during a staged rollout. Override
// each supported operation; the others fail explicitly, never fall back to CPU.
type UnsupportedOps struct{}

func (UnsupportedOps) MatMul(out, a, b *Buffer, m, n, k int, ta, tb bool) error {
	return fmt.Errorf("%w: matmul", ErrUnsupported)
}
func (UnsupportedOps) Add(out, a, b *Buffer, n int) error {
	return fmt.Errorf("%w: add", ErrUnsupported)
}
func (UnsupportedOps) Mul(out, a, b *Buffer, n int) error {
	return fmt.Errorf("%w: mul", ErrUnsupported)
}
func (UnsupportedOps) MulScalar(out, a *Buffer, n int, scalar float32) error {
	return fmt.Errorf("%w: mul_scalar", ErrUnsupported)
}
func (UnsupportedOps) GELU(out, a *Buffer, n int) error {
	return fmt.Errorf("%w: gelu", ErrUnsupported)
}
func (UnsupportedOps) Softmax(out, a *Buffer, rows, cols int) error {
	return fmt.Errorf("%w: softmax", ErrUnsupported)
}
func (UnsupportedOps) LayerNorm(out, a, w, b *Buffer, rows, cols int, eps float32) error {
	return fmt.Errorf("%w: layer_norm", ErrUnsupported)
}
func (UnsupportedOps) GroupNorm(out, a, w, b *Buffer, batch, frames, channels, groups int, eps float32) error {
	return fmt.Errorf("%w: group_norm", ErrUnsupported)
}
func (UnsupportedOps) Conv1d(out, x, w, bias *Buffer, options Conv1dOptions) error {
	return fmt.Errorf("%w: conv1d", ErrUnsupported)
}
