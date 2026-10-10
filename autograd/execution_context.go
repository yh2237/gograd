package autograd

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/yh2237/gograd/tensor"
)

// ExecutionContext owns a graph-recording scope. Tensors carry it through
// operations and views, so separate contexts can record/infer concurrently
// without a process-wide recording switch or goroutine-local state.
//
// The zero value records gradients. Do not copy a context after first use.
// NoGrad affects every operation on this context until the callback returns;
// use separate contexts (and separate model state) for independent execution.
// This does not manage CUDA threads/streams or make shared Tensor mutation safe.
type ExecutionContext struct {
	noGradDepth atomic.Int64
	optionsMu   sync.RWMutex
	options     ExecutionOptions
}

func NewExecutionContext() *ExecutionContext { return &ExecutionContext{} }

// Recording reports whether new results on this context record derivatives.
func (c *ExecutionContext) Recording() bool {
	if c == nil {
		panic("autograd: nil execution context")
	}
	return c.noGradDepth.Load() == 0
}

// NoGrad disables derivative recording only on this context. Nested scopes and
// panics restore the previous scope depth. Existing graphs can still Backward.
// Factory requiresGrad arguments are preserved, just as with package New.
func (c *ExecutionContext) NoGrad(fn func()) {
	if c == nil {
		panic("autograd: nil execution context")
	}
	c.noGradDepth.Add(1)
	defer c.noGradDepth.Add(-1)
	fn()
}

// New constructs a leaf on this context. BindModule associates parameters
// created by the existing layer constructors with the same context.
func (c *ExecutionContext) New(data []float32, shape []int, device tensor.Device, requiresGrad bool) (*Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("autograd: nil execution context")
	}
	v, err := New(data, shape, device, requiresGrad)
	if err == nil {
		v.execution = c
	}
	return v, err
}

func (c *ExecutionContext) Zeros(shape []int, device tensor.Device, requiresGrad bool) (*Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("autograd: nil execution context")
	}
	v, err := Zeros(shape, device, requiresGrad)
	if err == nil {
		v.execution = c
	}
	return v, err
}

func (c *ExecutionContext) checkBind(v *Tensor) error {
	if c == nil || v == nil {
		return fmt.Errorf("autograd: binding requires a context and a tensor")
	}
	if err := v.checkOpen(); err != nil {
		return err
	}
	if v.intermediate || v.ephemeral {
		return fmt.Errorf("autograd: only leaves can be bound to an execution context")
	}
	if v.execution != nil && v.execution != c {
		return fmt.Errorf("autograd: tensor already belongs to another execution context")
	}
	return nil
}

// Bind associates an existing leaf with this context without copying values,
// gradients or optimizer state. Bind before building graphs or starting workers;
// changing ownership of a leaf already in use is not supported.
func (c *ExecutionContext) Bind(v *Tensor) error {
	if err := c.checkBind(v); err != nil {
		return err
	}
	v.execution = c
	return nil
}

// BindModule associates every parameter and buffer in the module tree with
// this context. All leaves are validated before any ownership changes. Bind
// before Forward: parameter-only subexpressions such as embedding and weight
// views must run on the context too, not on the legacy recording scope.
func (c *ExecutionContext) BindModule(m *Module) error {
	if c == nil || m == nil {
		return fmt.Errorf("autograd: binding requires a context and a module")
	}
	var leaves []*Tensor
	seen := make(map[*Module]bool)
	var collect func(*Module) error
	collect = func(node *Module) error {
		if node == nil {
			return fmt.Errorf("autograd: nil child module")
		}
		if seen[node] {
			return nil
		}
		seen[node] = true
		for _, entries := range [][]Parameter{node.Parameters, node.Buffers} {
			for _, p := range entries {
				if err := c.checkBind(p.Value); err != nil {
					return fmt.Errorf("autograd: bind %s: %w", p.Name, err)
				}
				leaves = append(leaves, p.Value)
			}
		}
		for _, child := range node.Children {
			if err := collect(child.Module); err != nil {
				return err
			}
		}
		return nil
	}
	if err := collect(m); err != nil {
		return err
	}
	for _, v := range leaves {
		v.execution = c
	}
	return nil
}

// ExecutionContext returns this tensor's explicit scope, or nil for tensors
// using the compatibility API. Detach preserves the context but drops history.
func (t *Tensor) ExecutionContext() *ExecutionContext { return t.execution }

// Unbound tensors retain the old single-scope API. Explicit contexts never
// consult this default, so legacy NoGrad cannot switch off a bound model.
var defaultExecutionContext ExecutionContext

func mergeExecutionContext(context *ExecutionContext, v *Tensor) *ExecutionContext {
	if v == nil || v.execution == nil {
		return context
	}
	if context != nil && context != v.execution {
		panic("autograd: mixed execution contexts")
	}
	return v.execution
}

func graphRecording(parents []*Tensor, saved ...*Tensor) (*ExecutionContext, bool) {
	var context *ExecutionContext
	requiresGrad := false
	for _, p := range parents {
		context = mergeExecutionContext(context, p)
		requiresGrad = requiresGrad || p.RequiresGrad
	}
	for _, p := range saved {
		context = mergeExecutionContext(context, p)
	}
	if context == nil {
		return nil, requiresGrad && defaultExecutionContext.Recording()
	}
	return context, requiresGrad && context.Recording()
}
