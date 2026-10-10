package autograd

import "fmt"

// ExecutionOptions selects numerical/algorithmic policy for new forward ops.
// Explicit contexts default to FP32 and automatic attention selection, without
// inheriting the compatibility package globals.
type ExecutionOptions struct {
	// BF16Autocast uses BF16 CUDA GEMM operands with FP32 accumulation/master
	// values. CPU and currently unsupported operations (e.g. Conv2d) stay FP32.
	BF16Autocast bool
	// AttentionAlgorithm is auto, materialized or flash. Empty means auto.
	AttentionAlgorithm string
}

func (o ExecutionOptions) normalized() (ExecutionOptions, error) {
	if o.AttentionAlgorithm == "" {
		o.AttentionAlgorithm = "auto"
	}
	switch o.AttentionAlgorithm {
	case "auto", "materialized", "flash":
		return o, nil
	default:
		return o, fmt.Errorf("autograd: unknown attention algorithm %q", o.AttentionAlgorithm)
	}
}

// SetOptions changes policy for subsequent forwards. Already-created nodes
// keep their own backward policy. Set between forwards; changing policy during
// a multi-op Forward does not make the whole forward an atomic transaction.
func (c *ExecutionContext) SetOptions(options ExecutionOptions) error {
	if c == nil {
		return fmt.Errorf("autograd: nil execution context")
	}
	o, err := options.normalized()
	if err != nil {
		return err
	}
	c.optionsMu.Lock()
	c.options = o
	c.optionsMu.Unlock()
	return nil
}

// Options returns a copy of policy, suitable for passing to SetOptions.
func (c *ExecutionContext) Options() ExecutionOptions {
	if c == nil {
		panic("autograd: nil execution context")
	}
	c.optionsMu.RLock()
	o := c.options
	c.optionsMu.RUnlock()
	if o.AttentionAlgorithm == "" {
		o.AttentionAlgorithm = "auto"
	}
	return o
}

// Autocast temporarily enables/disables BF16 for this context. Nested scopes
// and panics restore the previous precision; attention selection is unaffected.
// Like NoGrad, this affects the entire context: do not overlap scopes from
// independent workers on the same context.
func (c *ExecutionContext) Autocast(bf16 bool, fn func()) {
	if c == nil {
		panic("autograd: nil execution context")
	}
	c.optionsMu.Lock()
	old := c.options.BF16Autocast
	c.options.BF16Autocast = bf16
	c.optionsMu.Unlock()
	defer func() {
		c.optionsMu.Lock()
		c.options.BF16Autocast = old
		c.optionsMu.Unlock()
	}()
	fn()
}

// Resolve once at forward entry and capture the returned value in backward.
// A bound op never reads the compatibility globals. Unbound constants may join
// a bound graph; parameters/parameter-only views should also be explicitly bound.
func executionOptions(inputs ...*Tensor) ExecutionOptions {
	var context *ExecutionContext
	for _, v := range inputs {
		context = mergeExecutionContext(context, v)
	}
	if context != nil {
		return context.Options()
	}
	return ExecutionOptions{BF16Autocast: BF16Autocast, AttentionAlgorithm: AttentionAlgorithm}
}
