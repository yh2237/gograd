package autograd

import "fmt"

// GradAccumulator spreads one optimizer update over several micro-batches. It
// scales each loss by 1/groups before Backward, so the accumulated gradient is
// the mean over the micro-batches of a group. The caller keeps the optimizer
// loop: zero gradients before the first micro-batch of a group, clip the
// accumulated gradient, and step when Backward reports that the group is done.
//
// A partial final group keeps the 1/groups scaling, so it forms a smaller
// effective batch. Callers that decide to step early call Reset, and a
// checkpoint should be taken only after a completed group because pending
// gradients live in the parameters' gradient buffers.
type GradAccumulator struct {
	groups  int
	pending int
}

// NewGradAccumulator creates an accumulator for groups micro-batches per update.
func NewGradAccumulator(groups int) (*GradAccumulator, error) {
	if groups < 1 {
		return nil, fmt.Errorf("autograd: gradient accumulation needs at least one micro-batch")
	}
	return &GradAccumulator{groups: groups}, nil
}

// Backward scales loss by 1/groups, back-propagates it, and reports whether this
// call completed a group. It consumes the loss graph, so the caller releases the
// loss afterwards as usual.
func (a *GradAccumulator) Backward(loss *Tensor) (bool, error) {
	if a == nil || a.groups < 1 {
		return false, fmt.Errorf("autograd: invalid gradient accumulator")
	}
	if loss == nil {
		return false, fmt.Errorf("autograd: nil accumulation loss")
	}
	scaled := MulScalar(loss, 1/float32(a.groups))
	err := scaled.Backward()
	scaled.ReleaseGraph()
	if err != nil {
		return false, err
	}
	a.pending++
	if a.pending >= a.groups {
		a.pending = 0
		return true, nil
	}
	return false, nil
}

// Pending reports how many micro-batches are accumulated since the last
// completed group.
func (a *GradAccumulator) Pending() int {
	if a == nil {
		return 0
	}
	return a.pending
}

// Groups reports the micro-batches per update the accumulator was created with.
func (a *GradAccumulator) Groups() int {
	if a == nil {
		return 0
	}
	return a.groups
}

// Reset forgets the pending micro-batches. Call it after stepping a partial
// group early, for example when an epoch ends before the group is complete.
func (a *GradAccumulator) Reset() {
	if a != nil {
		a.pending = 0
	}
}
