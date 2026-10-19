package gpu

// Synchronizer waits for prior backend work without downloading tensor values.
// Inference timing tools require it to measure completion, not just submission.
type Synchronizer interface {
	Synchronize() error
}
