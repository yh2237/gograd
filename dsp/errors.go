package dsp

import "fmt"

// Errors shared by the package. They are unexported because every caller
// either controls the transform length itself or reports a shape mismatch
// through its own domain error.
var (
	errNotPowerOfTwo = func(n int) error { return fmt.Errorf("dsp: length %d is not a power of two", n) }
)
