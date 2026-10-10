package autograd

import (
	"fmt"
	"math/rand"
	"slices"
	"sync"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func executionTensor(t *testing.T, c *ExecutionContext, values []float32, shape []int, device tensor.Device, grad bool) *Tensor {
	t.Helper()
	v, err := c.New(values, shape, device, grad)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(v.Close)
	return v
}

func TestExecutionContextScopes(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				defer ctx.Close()
			}
			// The zero-value context is usable and records by default.
			var c ExecutionContext
			x := executionTensor(t, &c, []float32{1, 2, 3, 4}, []int{2, 2}, device, true)
			if !c.Recording() || x.ExecutionContext() != &c {
				t.Fatal("context not attached")
			}
			trainLoss := Sum(Mul(x, x), 0, 1)
			defer trainLoss.ReleaseGraph()
			if !trainLoss.RequiresGrad || trainLoss.ExecutionContext() != &c {
				t.Fatal("training graph not recorded")
			}
			c.NoGrad(func() {
				if c.Recording() {
					t.Fatal("NoGrad did not enter")
				}
				// A pre-existing graph can Backward even inside NoGrad.
				if err := trainLoss.Backward(); err != nil {
					t.Fatal(err)
				}
				c.NoGrad(func() {
					v := Sum(Mul(x, x), 0, 1)
					defer v.ReleaseGraph()
					if v.RequiresGrad || v.backward != nil || v.backwardGPU != nil || v.ExecutionContext() != &c {
						t.Fatal("NoGrad recorded derivatives")
					}
					if err := v.Backward(); err == nil {
						t.Fatal("NoGrad backward accepted")
					}
				})
				if c.Recording() {
					t.Fatal("inner scope ended outer NoGrad")
				}
				panicked := false
				func() {
					defer func() { panicked = recover() != nil }()
					c.NoGrad(func() { panic("restore scope") })
				}()
				if !panicked || c.Recording() {
					t.Fatal("panic did not preserve outer scope")
				}
				// Explicit factory requiresGrad is not switched off by NoGrad.
				leaf := executionTensor(t, &c, []float32{2}, []int{}, device, true)
				if !leaf.RequiresGrad {
					t.Fatal("factory requiresGrad was changed")
				}
			})
			if !c.Recording() {
				t.Fatal("NoGrad did not restore")
			}
			conv2dClose(t, "existing backward", conv2dHost(t, x, true), []float32{2, 4, 6, 8}, 0, 0)
			z, err := c.Zeros([]int{2}, device, false)
			if err != nil {
				t.Fatal(err)
			}
			defer z.Close()
			if z.ExecutionContext() != &c {
				t.Fatal("Zeros context not attached")
			}
		})
	}
}

func TestExecutionContextViewsAndDetach(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				defer ctx.Close()
			}
			c := NewExecutionContext()
			x := executionTensor(t, c, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, true)
			view := Transpose(x, 0, 1)
			direct := view.Detach()
			defer direct.Close()
			if direct.ExecutionContext() != c || direct.RequiresGrad || len(direct.parents) != 0 {
				t.Fatal("view detach context/history")
			}
			conv2dClose(t, "view detach", conv2dHost(t, direct, false), []float32{1, 4, 2, 5, 3, 6}, 0, 0)
			materialized := view.Contiguous()
			if view.ExecutionContext() != c || materialized.ExecutionContext() != c {
				t.Fatal("view lost context")
			}
			detached := materialized.Detach()
			defer detached.Close()
			if detached.ExecutionContext() != c || detached.RequiresGrad || len(detached.parents) != 0 {
				t.Fatal("detach context/history")
			}
			materialized.ReleaseGraph()
			// Detach made independent storage. Graph release must not return that
			// new leaf's CPU slice to the intermediate pool.
			detached.ReleaseGraph()
			conv2dClose(t, "detached data", conv2dHost(t, detached, false), []float32{1, 4, 2, 5, 3, 6}, 0, 0)
			c.NoGrad(func() {
				v := Slice(x, 1, 0, 2)
				defer v.ReleaseGraph()
				if v.RequiresGrad || v.backward != nil || v.backwardGPU != nil || v.ExecutionContext() != c {
					t.Fatal("view ignored NoGrad")
				}
				copy := v.Detach()
				defer copy.Close()
				conv2dClose(t, "slice detach", conv2dHost(t, copy, false), []float32{1, 2, 4, 5}, 0, 0)
			})
			// View gradients still scatter to the bound leaf after scope exit.
			loss := Sum(Mul(Transpose(x, 0, 1), Transpose(x, 0, 1)), 0, 1)
			defer loss.ReleaseGraph()
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			conv2dClose(t, "view gradient", conv2dHost(t, x, true), []float32{2, 4, 6, 8, 10, 12}, 0, 0)
		})
	}
}

func TestExecutionContextBindModule(t *testing.T) {
	c, foreign := NewExecutionContext(), NewExecutionContext()
	w := Must([]float32{1, 2}, []int{2}, true)
	buffer := executionTensor(t, foreign, []float32{1}, []int{1}, tensor.CPU, false)
	m := Module{Parameters: []Parameter{{"weight", w}}, Children: []NamedModule{{"child", &Module{Buffers: []Parameter{{"buffer", buffer}}}}}}
	if err := c.BindModule(&m); err == nil {
		t.Fatal("foreign buffer accepted")
	}
	if w.ExecutionContext() != nil {
		t.Fatal("failed binding partially changed parameters")
	}
	m.Children[0].Module.Buffers[0].Value = Must([]float32{1}, []int{1}, false)
	if err := c.BindModule(&m); err != nil {
		t.Fatal(err)
	}
	if w.ExecutionContext() != c || m.Children[0].Module.Buffers[0].Value.ExecutionContext() != c {
		t.Fatal("module tree not bound")
	}
	if err := c.BindModule(&m); err != nil {
		t.Fatal("idempotent binding", err)
	}
	if err := foreign.Bind(w); err == nil {
		t.Fatal("reassigned bound leaf")
	}
	y := Mul(w, w)
	defer y.ReleaseGraph()
	if err := c.Bind(y); err == nil {
		t.Fatal("bound an existing graph result")
	}
	if err := c.Bind(nil); err == nil {
		t.Fatal("nil tensor accepted")
	}
	if err := c.BindModule(nil); err == nil {
		t.Fatal("nil module accepted")
	}
}

func TestExecutionContextMixedGraphs(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				defer ctx.Close()
			}
			c1, c2 := NewExecutionContext(), NewExecutionContext()
			a := executionTensor(t, c1, []float32{1, 2, 3, 4}, []int{2, 2}, device, true)
			b := executionTensor(t, c2, []float32{1, 2, 3, 4}, []int{2, 2}, device, true)
			neutral := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, device, false)
			for name, fn := range map[string]func(){
				"add": func() { Add(a, b) }, "matmul": func() { MatMul(a, b) },
				"concat": func() { Concat(0, neutral, a, b) },
			} {
				t.Run(name, func(t *testing.T) {
					before := cuda.MemoryStats().LiveBytes
					panicked := false
					func() { defer func() { panicked = recover() != nil }(); fn() }()
					if !panicked {
						t.Fatal("mixed graphs were accepted")
					}
					if device == tensor.CUDA && before != cuda.MemoryStats().LiveBytes {
						t.Fatal("mixed-graph rejection leaked device memory")
					}
				})
			}
			// Unbound constants remain interoperable with a bound graph.
			loss := Sum(Add(a, neutral), 0, 1)
			defer loss.ReleaseGraph()
			if loss.ExecutionContext() != c1 {
				t.Fatal("constant dropped graph context")
			}
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			conv2dClose(t, "constant join", conv2dHost(t, a, true), []float32{1, 1, 1, 1}, 0, 0)
		})
	}
}

// The legacy NoGrad is deliberately held open for the entire test. Bound
// parameter-only subexpressions (Transpose(weight), Add(bias)) must not consult
// it. Independent train and inference workers then share no mutable model state.
func TestExecutionContextCPUConcurrency(t *testing.T) {
	legacyEntered, legacyRelease, legacyDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() { defer close(legacyDone); NoGrad(func() { close(legacyEntered); <-legacyRelease }) }()
	<-legacyEntered
	defer func() { close(legacyRelease); <-legacyDone }()
	const workers = 8
	var wg sync.WaitGroup
	errors := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errors <- fmt.Errorf("worker %d panicked: %v", worker, r)
				}
			}()
			c := NewExecutionContext()
			l, err := NewLinearLayer(2, 2, tensor.CPU, rand.New(rand.NewSource(int64(worker))))
			if err != nil {
				errors <- err
				return
			}
			defer l.Weight.Close()
			defer l.Bias.Close()
			if err = c.BindModule(l.StateModule()); err != nil {
				errors <- err
				return
			}
			if err = l.Weight.CopyFrom([]float32{1, 2, 3, 4}); err != nil {
				errors <- err
				return
			}
			if err = l.Bias.CopyFrom([]float32{.1, .2}); err != nil {
				errors <- err
				return
			}
			x, err := c.New([]float32{.2, .3}, []int{1, 2}, tensor.CPU, true)
			if err != nil {
				errors <- err
				return
			}
			defer x.Close()
			for step := 0; step < 30; step++ {
				x.ZeroGrad()
				l.Weight.ZeroGrad()
				l.Bias.ZeroGrad()
				if worker%2 == 0 {
					loss := Sum(l.Forward(x), 0, 1)
					if !loss.RequiresGrad || loss.ExecutionContext() != c {
						loss.ReleaseGraph()
						errors <- fmt.Errorf("worker %d lost recording", worker)
						return
					}
					if err = loss.Backward(); err != nil {
						loss.ReleaseGraph()
						errors <- err
						return
					}
					loss.ReleaseGraph()
					if !slices.Equal(x.Grad, []float32{4, 6}) || !slices.Equal(l.Bias.Grad, []float32{1, 1}) || !slices.Equal(l.Weight.Grad, []float32{.2, .3, .2, .3}) {
						errors <- fmt.Errorf("worker %d incorrect gradients: x=%v w=%v b=%v", worker, x.Grad, l.Weight.Grad, l.Bias.Grad)
						return
					}
				} else {
					var failure error
					c.NoGrad(func() {
						loss := Sum(l.Forward(x), 0, 1)
						defer loss.ReleaseGraph()
						if loss.RequiresGrad || loss.backward != nil || loss.ExecutionContext() != c {
							failure = fmt.Errorf("worker %d inference recorded", worker)
						}
					})
					if failure != nil {
						errors <- failure
						return
					}
					if !c.Recording() || x.Grad != nil || l.Weight.Grad != nil || l.Bias.Grad != nil {
						errors <- fmt.Errorf("worker %d inference mutated gradients/scope", worker)
						return
					}
				}
			}
		}(worker)
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestExecutionContextIndependentScopes(t *testing.T) {
	train, infer := NewExecutionContext(), NewExecutionContext()
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() { defer close(done); infer.NoGrad(func() { close(entered); <-release }) }()
	<-entered
	defer func() { close(release); <-done }()
	if !train.Recording() || infer.Recording() {
		t.Fatal("scopes not independent")
	}
	x := executionTensor(t, train, []float32{3}, []int{}, tensor.CPU, true)
	loss := Mul(x, x)
	defer loss.ReleaseGraph()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(x.Grad, []float32{6}) {
		t.Fatal(x.Grad)
	}
}
