package gpu

// ExactGELUBackend optionally provides the erf-based GELU requested by Tensor
// APIs with approx=false. Backend.GELU always means the tanh approximation.
type ExactGELUBackend interface {
	GELUExact(out, a *Buffer, n int) error
}
