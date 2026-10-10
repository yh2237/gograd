package autograd

import (
	"sync"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestBatchNormSelectiveGradientsAndClosedInput(t *testing.T) {
	f := readBatchNormFixtures(t).Cases[0]
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for selected, name := range []string{"input", "weight", "bias"} {
			t.Run(name, func(t *testing.T) {
				pool2dContext(t, device)
				x := conv2dTestTensor(t, f.X, f.XShape, device, selected == 0)
				w := conv2dTestTensor(t, f.W, []int{3}, device, selected == 1)
				b := conv2dTestTensor(t, f.B, []int{3}, device, selected == 2)
				up := conv2dTestTensor(t, f.Upstream, f.XShape, device, false)
				loss := weightedSum(BatchNorm(x, w, b, nil, nil, f.Options), up)
				defer loss.ReleaseGraph()
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				for i, param := range []*Tensor{x, w, b} {
					values := conv2dHost(t, param, true)
					if i == selected {
						conv2dClose(t, "selected VJP", values, [][]float32{f.DX, f.DW, f.DB}[i], 1e-5, 8e-5)
					} else if len(values) != 0 {
						t.Fatal("frozen BatchNorm operand acquired a gradient")
					}
				}
			})
		}
		base := conv2dTestTensor(t, f.X, f.XShape, device, true)
		view := Reshape(base, base.Shape...)
		view.RetainGrad()
		defer view.Close()
		w := conv2dTestTensor(t, f.W, []int{3}, device, true)
		b := conv2dTestTensor(t, f.B, []int{3}, device, true)
		up := conv2dTestTensor(t, f.Upstream, f.XShape, device, false)
		loss := weightedSum(BatchNorm(view, w, b, nil, nil, f.Options), up)
		defer loss.ReleaseGraph()
		base.Close()
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "closed input VJP", conv2dHost(t, view, true), f.DX, 1e-5, 8e-5)
	})
}

func TestBatchNormCPUIndependentWorkers(t *testing.T) {
	const workers, steps = 6, 20
	var wg sync.WaitGroup
	// Explicit contexts remain independent even with compatibility NoGrad held.
	NoGrad(func() {
		for worker := 0; worker < workers; worker++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				layer, err := NewBatchNormLayer(2, BatchNormLayerOptions{}, tensor.CPU)
				if err != nil {
					t.Error(err)
					return
				}
				defer layer.Close()
				context := NewExecutionContext()
				if err := context.BindModule(layer.StateModule()); err != nil {
					t.Error(err)
					return
				}
				x, err := context.New(policyValues(24, float64(worker)), []int{3, 2, 4}, tensor.CPU, true)
				if err != nil {
					t.Error(err)
					return
				}
				defer x.Close()
				up, err := context.New(policyValues(24, .7), x.Shape, tensor.CPU, false)
				if err != nil {
					t.Error(err)
					return
				}
				defer up.Close()
				for step := 0; step < steps; step++ {
					x.ZeroGrad()
					layer.Weight.ZeroGrad()
					layer.Bias.ZeroGrad()
					if worker%2 == 0 {
						context.NoGrad(func() {
							y := layer.Forward(x)
							if y.RequiresGrad {
								t.Error("independent NoGrad recorded BatchNorm")
							}
							y.ReleaseGraph()
						})
					} else {
						loss := weightedSum(layer.Forward(x), up)
						if err := loss.Backward(); err != nil {
							t.Error(err)
						}
						loss.ReleaseGraph()
						if len(x.Grad) != x.Numel() {
							t.Error("compatibility scope disabled bound worker")
						}
					}
				}
				if layer.NumBatchesTracked.Data[0] != steps {
					t.Error("BatchNorm workers shared or lost state")
				}
				layer.Train(false)
				context.NoGrad(func() { y := layer.Forward(x); y.ReleaseGraph() })
				if layer.NumBatchesTracked.Data[0] != steps {
					t.Error("evaluation updated BatchNorm state")
				}
			}(worker)
		}
		wg.Wait()
	})
}
