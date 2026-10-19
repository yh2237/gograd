package gpu

// FusedBiasBackend optionally avoids intermediate tensors for two common
// inference operations. Buffers are contiguous: a, residual and out contain
// rows*cols elements, and bias contains cols elements. cols must be positive;
// rows may be zero. The output must not alias any input; inputs may alias each
// other. BiasGELU uses the erf formula, never the tanh approximation.
//
// ErrUnsupported allows the caller to compose the existing GPU operations.
// Other errors must be propagated, not retried or evaluated on the CPU.
type FusedBiasBackend interface {
	BiasGELU(out, a, bias *Buffer, rows, cols int) error
	// BiasResidual computes (a + bias) + residual, preserving addition order.
	BiasResidual(out, a, bias, residual *Buffer, rows, cols int) error
}
