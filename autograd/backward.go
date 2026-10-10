package autograd

import (
	"fmt"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// BackwardOptions controls saved derivative history, independently of retaining
// a tensor's values or gradient. Default Backward consumes the history.
type BackwardOptions struct {
	RetainGraph bool
}

// BackwardWithOptions permits another backward through shared history when
// RetainGraph is true. Leaf gradients accumulate; intermediate gradients are
// fresh per pass so a repeated backward never propagates an old adjoint twice.
func (t *Tensor) BackwardWithOptions(options BackwardOptions) error {
	if err := t.checkOpen(); err != nil {
		return err
	}
	if t.Numel() != 1 {
		return fmt.Errorf("autograd: backward requires scalar")
	}
	if !t.RequiresGrad {
		return fmt.Errorf("autograd: backward requires a gradient-recorded tensor")
	}
	var topo []*Tensor
	seen := make(map[*Tensor]bool)
	var visit func(*Tensor) error
	visit = func(v *Tensor) error {
		if seen[v] || !v.RequiresGrad {
			return nil
		}
		seen[v] = true
		// An explicitly closed ancestor remains readable internally while this
		// graph has a dependency on it. Disposed storage cannot be used.
		if v.disposed.Load() || v.storage == nil {
			return fmt.Errorf("autograd: backward references released storage")
		}
		if v.intermediate {
			if v.historyReleased {
				return fmt.Errorf("autograd: graph history was released; retain it on the first backward or rebuild the graph")
			}
			if v.storage.version.Load() != v.outputVersion {
				return fmt.Errorf("autograd: saved output was modified after forward")
			}
			for i, p := range v.parents {
				if p.storage == nil || p.storage.version.Load() != v.savedVersions[i] {
					return fmt.Errorf("autograd: saved input was modified or released after forward")
				}
			}
		}
		for _, p := range v.gradParents {
			if err := visit(p); err != nil {
				return err
			}
		}
		topo = append(topo, v)
		return nil
	}
	// Validate the entire gradient graph before changing any gradients.
	if err := visit(t); err != nil {
		return err
	}
	oldCPU := make(map[*Tensor][]float32)
	oldGPU := make(map[*Tensor]*cuda.Buffer)
	for _, v := range topo {
		v.inBackward.Store(true)
		if !v.intermediate {
			continue
		}
		if v.retain {
			oldCPU[v], oldGPU[v] = v.Grad, v.gradBuf
			v.Grad, v.gradBuf = nil, nil
		} else {
			v.clearIntermediateGradient()
		}
	}
	// Non-gradient branches still own values/auxiliaries. Mark their lifetimes
	// before CUDA starts dropping completed nodes' edges. Gradient nodes are
	// protected by inBackward until their own callback has finished.
	if t.Device == tensor.CUDA && !options.RetainGraph {
		for _, v := range t.collectLifetime() {
			if v.intermediate || v.ephemeral {
				v.releaseRequested.Store(true)
			}
		}
	}
	defer func() {
		for _, b := range oldGPU {
			if b != nil {
				b.Free()
			}
		}
		for _, v := range topo {
			v.inBackward.Store(false)
			v.maybeDispose()
		}
	}()
	if t.Device == tensor.CUDA {
		if t.intermediate {
			setOne(t.ensureGradGPU())
		} else {
			one := mustAlloc(1)
			setOne(one)
			addDevice(t.ensureGradGPU(), one, 1)
			one.Free()
		}
	} else {
		t.addGrad([]float32{1})
	}
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		if v.Device == tensor.CUDA {
			if v.backwardGPU != nil && v.gradBuf != nil {
				v.backwardGPU(v.gradBuf)
			}
		} else if v.backward != nil && v.Grad != nil {
			v.backward(v.Grad)
		}
		if v.intermediate {
			if v.retain {
				if previous := oldCPU[v]; previous != nil {
					if v.Grad == nil {
						v.Grad = previous
					} else {
						for j, value := range previous {
							v.Grad[j] += value
						}
					}
				}
				if previous := oldGPU[v]; previous != nil {
					if v.gradBuf == nil {
						v.gradBuf = previous
					} else {
						addDevice(v.gradBuf, previous, v.Numel())
						previous.Free()
					}
					delete(oldGPU, v)
				}
			} else {
				v.clearIntermediateGradient()
			}
			if !options.RetainGraph {
				v.releaseSaved()
			}
		}
		if v.Device == tensor.CUDA && !options.RetainGraph && (v.intermediate || v.ephemeral) {
			v.releaseRequested.Store(true)
		}
		v.inBackward.Store(false)
		v.maybeDispose()
	}
	if t.Device == tensor.CUDA && !options.RetainGraph {
		t.ReleaseGraph()
	}
	return nil
}

func (t *Tensor) clearIntermediateGradient() {
	t.Grad = nil
	if t.gradBuf != nil {
		t.gradBuf.Free()
		t.gradBuf = nil
	}
}
