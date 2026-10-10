// Command cnn-train trains a small NCHW image classifier without external data
// or Python. It optionally saves weights and verifies inference after reload.
package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
)

type classifier struct {
	module        autograd.Module
	first, second *autograd.Conv2dLayer
	head          *autograd.LinearLayer
}

func newClassifier(device tensor.Device) (*classifier, error) {
	m := &classifier{}
	rng := rand.New(rand.NewSource(7))
	opts := autograd.Conv2dOptions{Padding: [2]int{1, 1}}
	var err error
	if m.first, err = autograd.NewConv2dLayer(1, 8, [2]int{3, 3}, opts, true, device, rng); err != nil {
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "conv1", Module: m.first.StateModule()})
	if m.second, err = autograd.NewConv2dLayer(8, 8, [2]int{3, 3}, opts, true, device, rng); err != nil {
		m.close()
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "conv2", Module: m.second.StateModule()})
	if m.head, err = autograd.NewLinearLayer(8, 2, device, rng); err != nil {
		m.close()
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "head", Module: m.head.StateModule()})
	return m, nil
}

func (m *classifier) forward(x *autograd.Tensor) *autograd.Tensor {
	h := autograd.ReLU(m.first.Forward(x))
	h = autograd.MaxPool2d(h, autograd.MaxPool2dOptions{KernelSize: [2]int{2, 2}})
	h = autograd.ReLU(m.second.Forward(h))
	// Adaptive pooling keeps the classifier independent of spatial dimensions.
	h = autograd.AdaptiveAvgPool2d(h, [2]int{1, 1})
	return m.head.Forward(autograd.Reshape(h, h.Shape[0], h.Shape[1]))
}

func (m *classifier) close() {
	for _, p := range m.module.NamedParameters() {
		p.Value.Close()
	}
}

// Alternating vertical/horizontal stripes with shifted phase and additive
// noise. Both classes have comparable mean brightness: classification requires
// learning spatial filters rather than a threshold on the raw image average.
func stripeImages() ([]float32, []int) {
	const batch, size = 32, 8
	rng := rand.New(rand.NewSource(19))
	x, labels := make([]float32, batch*size*size), make([]int, batch)
	for n := range labels {
		labels[n] = n % 2
		for y := 0; y < size; y++ {
			for z := 0; z < size; z++ {
				axis := z
				if labels[n] == 1 {
					axis = y
				}
				x[(n*size+y)*size+z] = float32((axis+n/2)%2) + float32(rng.NormFloat64()*.05)
			}
		}
	}
	return x, labels
}

type trainingStats struct {
	initialLoss, finalLoss, accuracy float32
	reloadMaxAbs                     float64
}

func train(device tensor.Device, steps int, output string, log io.Writer) (trainingStats, error) {
	var stats trainingStats
	if steps < 1 || (device != tensor.CPU && device != tensor.CUDA) {
		return stats, fmt.Errorf("cnn-train: positive steps and cpu/cuda device required")
	}
	if device == tensor.CUDA {
		ctx, err := autograd.NewCUDAContext()
		if err != nil {
			return stats, err
		}
		defer ctx.Close()
	}
	m, err := newClassifier(device)
	if err != nil {
		return stats, err
	}
	defer m.close()
	execution := autograd.NewExecutionContext()
	if err := execution.BindModule(&m.module); err != nil {
		return stats, err
	}
	m.module.Train(true)
	opt := autograd.NewAdamW(m.module.NamedParameters(), .01, .001)
	defer opt.Close()
	images, labels := stripeImages()
	x, err := execution.New(images, []int{len(labels), 1, 8, 8}, device, false)
	if err != nil {
		return stats, err
	}
	defer x.Close()
	for step := 0; step < steps; step++ {
		start := time.Now()
		opt.ZeroGrad()
		loss := autograd.CrossEntropy(m.forward(x), labels, -1)
		v, err := loss.ToHost()
		if err != nil {
			loss.ReleaseGraph()
			return stats, err
		}
		if step == 0 {
			stats.initialLoss = v[0]
		}
		if err = loss.Backward(); err != nil {
			loss.ReleaseGraph()
			return stats, err
		}
		opt.Step()
		loss.ReleaseGraph()
		if step == 0 || (step+1)%10 == 0 || step+1 == steps {
			fmt.Fprintf(log, "device=%s step=%d loss=%.6f duration=%s\n", device, step+1, v[0], time.Since(start))
		}
	}
	m.module.Train(false)
	var logits *autograd.Tensor
	execution.NoGrad(func() { logits = m.forward(x) })
	defer logits.ReleaseGraph()
	values, err := logits.ToHost()
	if err != nil {
		return stats, err
	}
	var final *autograd.Tensor
	execution.NoGrad(func() { final = autograd.CrossEntropy(logits, labels, -1) })
	v, err := final.ToHost()
	if err != nil {
		final.ReleaseGraph()
		return stats, err
	}
	stats.finalLoss = v[0]
	// final owns the logits graph too; values were downloaded above.
	final.ReleaseGraph()
	correct := 0
	for i, want := range labels {
		got := 0
		if values[i*2+1] > values[i*2] {
			got = 1
		}
		if got == want {
			correct++
		}
	}
	stats.accuracy = float32(correct) / float32(len(labels))
	if output != "" {
		if err := m.module.SaveSafeTensors(output); err != nil {
			return stats, err
		}
		loaded, err := newClassifier(device)
		if err != nil {
			return stats, err
		}
		defer loaded.close()
		if err := execution.BindModule(&loaded.module); err != nil {
			return stats, err
		}
		if err = loaded.module.LoadSafeTensors(output); err != nil {
			return stats, err
		}
		original, restoredState := m.module.StateDict(), loaded.module.StateDict()
		for name, want := range original {
			got := restoredState[name]
			if len(got) != len(want) {
				return stats, fmt.Errorf("cnn-train: checkpoint shape changed for %s", name)
			}
			for i, v := range want {
				if got[i] != v {
					return stats, fmt.Errorf("cnn-train: checkpoint changed %s[%d]", name, i)
				}
			}
		}
		var restored *autograd.Tensor
		execution.NoGrad(func() { restored = loaded.forward(x) })
		defer restored.ReleaseGraph()
		check, err := restored.ToHost()
		if err != nil {
			return stats, err
		}
		for i := range check {
			stats.reloadMaxAbs = math.Max(stats.reloadMaxAbs, math.Abs(float64(check[i]-values[i])))
		}
		// CUDA spatial reductions can accumulate in a different order on a
		// subsequent evaluation; checkpoint tensors themselves match exactly.
		if stats.reloadMaxAbs > 1e-5 {
			return stats, fmt.Errorf("cnn-train: checkpoint changed logits by %g", stats.reloadMaxAbs)
		}
		fmt.Fprintf(log, "saved=%s reload_max_abs=%.9g\n", output, stats.reloadMaxAbs)
	}
	fmt.Fprintf(log, "initial_loss=%.6f final_loss=%.6f accuracy=%.1f%%\n", stats.initialLoss, stats.finalLoss, 100*stats.accuracy)
	return stats, nil
}

func main() {
	device := flag.String("device", "cpu", "cpu or cuda")
	steps := flag.Int("steps", 60, "training updates")
	out := flag.String("out", "", "optional safetensors checkpoint (existing parent directory)")
	flag.Parse()
	if _, err := train(tensor.Device(*device), *steps, *out, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
