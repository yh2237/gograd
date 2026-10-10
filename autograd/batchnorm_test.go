package autograd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type batchNormFixture struct {
	Name                             string
	XShape                           []int `json:"x_shape"`
	Options                          BatchNormOptions
	X, W, B, Y, Upstream, DX, DW, DB []float32
	InitialMean                      []float32 `json:"initial_mean"`
	InitialVar                       []float32 `json:"initial_var"`
	RunningMean                      []float32 `json:"running_mean"`
	RunningVar                       []float32 `json:"running_var"`
}
type batchNormStep struct {
	Training                          bool
	NoGrad                            bool  `json:"no_grad"`
	XShape                            []int `json:"x_shape"`
	X, Y, Upstream, DX, DW, DB, Count []float32
	RunningMean                       []float32 `json:"running_mean"`
	RunningVar                        []float32 `json:"running_var"`
}
type batchNormSequence struct {
	Name                 string
	Cumulative, Disabled bool
	ChannelsLast         bool `json:"channels_last"`
	W, B                 []float32
	Steps                []batchNormStep
}
type batchNormFixtures struct {
	Cases     []batchNormFixture
	Sequences []batchNormSequence
}

func readBatchNormFixtures(t *testing.T) batchNormFixtures {
	t.Helper()
	data, err := os.ReadFile("../testdata/batchnorm.json")
	if err != nil {
		t.Fatal(err)
	}
	var f batchNormFixtures
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) == 0 || len(f.Sequences) == 0 {
		t.Fatal("empty BatchNorm fixtures")
	}
	return f
}

func TestBatchNormPyTorchParity(t *testing.T) {
	fixtures := readBatchNormFixtures(t)
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, f := range fixtures.Cases {
			t.Run(f.Name, func(t *testing.T) {
				pool2dContext(t, device)
				x := conv2dTestTensor(t, f.X, f.XShape, device, true)
				c := f.XShape[1]
				if f.Options.ChannelsLast {
					c = f.XShape[len(f.XShape)-1]
				}
				vector := func(values []float32, grad bool) *Tensor {
					if values == nil {
						return nil
					}
					return conv2dTestTensor(t, values, []int{c}, device, grad)
				}
				w, b, mean, variance := vector(f.W, true), vector(f.B, true), vector(f.InitialMean, false), vector(f.InitialVar, false)
				EnableGPUProfile(device == tensor.CUDA)
				defer EnableGPUProfile(false)
				y := BatchNorm(x, w, b, mean, variance, f.Options)
				if !slices.Equal(y.Shape, f.XShape) {
					t.Fatalf("shape %v", y.Shape)
				}
				if device == tensor.CUDA && (y.Data != nil || y.Buffer() == nil) {
					t.Fatal("BatchNorm output left CUDA")
				}
				conv2dClose(t, "BatchNorm output", conv2dHost(t, y, false), f.Y, 4e-6, 4e-5)
				if mean != nil {
					conv2dClose(t, "running mean", conv2dHost(t, mean, false), f.RunningMean, 2e-6, 2e-5)
					conv2dClose(t, "running variance", conv2dHost(t, variance, false), f.RunningVar, 2e-6, 2e-5)
				}
				up := conv2dTestTensor(t, f.Upstream, f.XShape, device, false)
				loss := weightedSum(y, up)
				defer loss.ReleaseGraph()
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "input VJP", conv2dHost(t, x, true), f.DX, 1e-5, 8e-5)
				if w != nil {
					conv2dClose(t, "weight VJP", conv2dHost(t, w, true), f.DW, 1e-5, 5e-5)
				}
				if b != nil {
					conv2dClose(t, "bias VJP", conv2dHost(t, b, true), f.DB, 1e-5, 5e-5)
				}
				if device == tensor.CUDA {
					profile := GPUProfile()
					for _, key := range []string{"bn_stats", "bn_f", "bn_grad_stats", "bn_dx"} {
						if profile[key].Count == 0 {
							t.Fatalf("missing CUDA %s", key)
						}
					}
				}
			})
		}
	})
}

func TestBatchNormStateSequenceAndReload(t *testing.T) {
	fixtures := readBatchNormFixtures(t)
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, f := range fixtures.Sequences {
			t.Run(f.Name, func(t *testing.T) {
				pool2dContext(t, device)
				momentum := float32(.25)
				opts := BatchNormLayerOptions{Momentum: &momentum, CumulativeMomentum: f.Cumulative, DisableRunningStats: f.Disabled, ChannelsLast: f.ChannelsLast}
				layer, err := NewBatchNormLayer(2, opts, device)
				if err != nil {
					t.Fatal(err)
				}
				defer layer.Close()
				if err := layer.Weight.CopyFrom(f.W); err != nil {
					t.Fatal(err)
				}
				if err := layer.Bias.CopyFrom(f.B); err != nil {
					t.Fatal(err)
				}
				root := Module{Children: []NamedModule{{Name: "norm", Module: layer.StateModule()}}}
				context := NewExecutionContext()
				if err := context.BindModule(&root); err != nil {
					t.Fatal(err)
				}
				for _, step := range f.Steps {
					root.Train(step.Training)
					if layer.Training != step.Training {
						t.Fatal("recursive mode propagation failed")
					}
					layer.Weight.ZeroGrad()
					layer.Bias.ZeroGrad()
					x := executionTensor(t, context, step.X, step.XShape, device, true)
					var y *Tensor
					if step.NoGrad {
						context.NoGrad(func() { y = layer.Forward(x) })
					} else {
						y = layer.Forward(x)
					}
					if y.ExecutionContext() != context {
						t.Fatal("BatchNorm lost context")
					}
					conv2dClose(t, "state sequence output", conv2dHost(t, y, false), step.Y, 5e-6, 5e-5)
					if !f.Disabled {
						conv2dClose(t, "sequence running mean", conv2dHost(t, layer.RunningMean, false), step.RunningMean, 3e-6, 3e-5)
						conv2dClose(t, "sequence running variance", conv2dHost(t, layer.RunningVar, false), step.RunningVar, 3e-6, 3e-5)
						conv2dClose(t, "sequence batch counter", conv2dHost(t, layer.NumBatchesTracked, false), step.Count, 0, 0)
					}
					if step.NoGrad {
						if y.RequiresGrad || y.backward != nil || y.backwardGPU != nil {
							t.Fatal("NoGrad recorded BatchNorm history")
						}
						y.ReleaseGraph()
					} else {
						up := executionTensor(t, context, step.Upstream, step.XShape, device, false)
						loss := weightedSum(y, up)
						if err := loss.Backward(); err != nil {
							t.Fatal(err)
						}
						conv2dClose(t, "sequence input gradient", conv2dHost(t, x, true), step.DX, 2e-5, 8e-5)
						conv2dClose(t, "sequence weight gradient", conv2dHost(t, layer.Weight, true), step.DW, 2e-5, 5e-5)
						conv2dClose(t, "sequence bias gradient", conv2dHost(t, layer.Bias, true), step.DB, 2e-5, 5e-5)
						loss.ReleaseGraph()
						up.Close()
					}
					x.Close()
				}
				file := filepath.Join(t.TempDir(), "norm.safetensors")
				if err := root.SaveSafeTensors(file); err != nil {
					t.Fatal(err)
				}
				restored, err := NewBatchNormLayer(2, opts, device)
				if err != nil {
					t.Fatal(err)
				}
				defer restored.Close()
				loaded := Module{Children: []NamedModule{{Name: "norm", Module: restored.StateModule()}}}
				if err := loaded.LoadSafeTensors(file); err != nil {
					t.Fatal(err)
				}
				for name, want := range root.StateDict() {
					conv2dClose(t, name, loaded.StateDict()[name], want, 0, 0)
				}
				if len(root.NamedParameters()) != 2 {
					t.Fatal("running buffers entered optimizer parameters")
				}
				root.Train(false)
				loaded.Train(false)
				last := f.Steps[len(f.Steps)-1]
				x := executionTensor(t, context, last.X, last.XShape, device, false)
				if err := context.BindModule(&loaded); err != nil {
					t.Fatal(err)
				}
				context.NoGrad(func() {
					a, b := layer.Forward(x), restored.Forward(x)
					defer a.ReleaseGraph()
					defer b.ReleaseGraph()
					conv2dClose(t, "reloaded inference", conv2dHost(t, b, false), conv2dHost(t, a, false), 1e-6, 1e-6)
				})
			})
		}
	})
}

func TestBatchNormViewsAndStatisticSnapshots(t *testing.T) {
	f := readBatchNormFixtures(t).Cases[2]
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		context := NewExecutionContext()
		layer, err := NewBatchNormLayer(3, BatchNormLayerOptions{}, device)
		if err != nil {
			t.Fatal(err)
		}
		defer layer.Close()
		if err := context.BindModule(layer.StateModule()); err != nil {
			t.Fatal(err)
		}
		if err := layer.Weight.CopyFrom(f.W); err != nil {
			t.Fatal(err)
		}
		if err := layer.Bias.CopyFrom(f.B); err != nil {
			t.Fatal(err)
		}
		values, shape := conv2dTransposedStorage(f.X, f.XShape)
		base := executionTensor(t, context, values, shape, device, true)
		x := Transpose(base, 2, 3)
		up := executionTensor(t, context, f.Upstream, f.XShape, device, false)
		first := weightedSum(layer.Forward(x), up)
		defer first.ReleaseGraph()
		// A second forward updates running state before the first backward.
		second := weightedSum(layer.Forward(x), up)
		defer second.ReleaseGraph()
		layer.Train(false)
		if err := first.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		if err := second.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		if err := first.Backward(); err != nil {
			t.Fatal(err)
		}
		want, _ := conv2dTransposedStorage(f.DX, f.XShape)
		for i := range want {
			want[i] *= 3
		}
		conv2dClose(t, "shared snapshot gradient", conv2dHost(t, base, true), want, 3e-5, 1e-4)
		layer.Train(false)
		input := executionTensor(t, context, f.X, f.XShape, device, true)
		y := layer.Forward(input)
		// Eval reads a private snapshot, not a mutable running-buffer edge.
		layer.RunningMean.Close()
		layer.RunningVar.Close()
		loss := Sum(y, 0, 1, 2, 3)
		defer loss.ReleaseGraph()
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		if len(conv2dHost(t, input, true)) != len(f.X) {
			t.Fatal("closed running state broke backward")
		}
	})
}

func TestBatchNormFiniteDifferences(t *testing.T) {
	f := readBatchNormFixtures(t).Cases[0]
	x := conv2dTestTensor(t, f.X, f.XShape, tensor.CPU, true)
	w := conv2dTestTensor(t, f.W, []int{3}, tensor.CPU, true)
	b := conv2dTestTensor(t, f.B, []int{3}, tensor.CPU, true)
	up := conv2dTestTensor(t, f.Upstream, f.XShape, tensor.CPU, false)
	forward := func() *Tensor { return BatchNorm(x, w, b, nil, nil, f.Options) }
	loss := weightedSum(forward(), up)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	loss.ReleaseGraph()
	evaluate := func() float32 {
		var out float32
		NoGrad(func() {
			y := forward()
			for i, v := range y.Data {
				out += v * f.Upstream[i]
			}
			y.ReleaseGraph()
		})
		return out
	}
	for _, param := range []*Tensor{x, w, b} {
		values := conv2dHost(t, param, false)
		grad := conv2dHost(t, param, true)
		for i := range values {
			plus := append([]float32(nil), values...)
			plus[i] += .001
			if err := param.CopyFrom(plus); err != nil {
				t.Fatal(err)
			}
			a := evaluate()
			plus[i] -= .002
			if err := param.CopyFrom(plus); err != nil {
				t.Fatal(err)
			}
			z := evaluate()
			if math.Abs(float64((a-z)/.002-grad[i])) > 6e-4 {
				t.Fatalf("finite difference[%d] %g want %g", i, (a-z)/.002, grad[i])
			}
			if err := param.CopyFrom(values); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestBatchNormValidationAndDefaults(t *testing.T) {
	l, err := NewBatchNormLayer(3, BatchNormLayerOptions{}, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if l.Eps != 1e-5 || l.Momentum != .1 || !l.Training || len(l.Module.Parameters) != 2 || len(l.Module.Buffers) != 3 {
		t.Fatal("incorrect default layer configuration")
	}
	disabled, err := NewBatchNormLayer(3, BatchNormLayerOptions{DisableAffine: true, DisableRunningStats: true}, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	defer disabled.Close()
	if len(disabled.Module.StateDict()) != 0 {
		t.Fatal("disabled state was registered")
	}
	zero := float32(0)
	frozen, err := NewBatchNormLayer(3, BatchNormLayerOptions{Momentum: &zero}, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	if frozen.Momentum != 0 {
		t.Fatal("explicit zero momentum ignored")
	}
	x := conv2dTestTensor(t, policyValues(12, .3), []int{4, 3}, tensor.CPU, true)
	single := conv2dTestTensor(t, []float32{1, 2, 3}, []int{1, 3}, tensor.CPU, true)
	wrong := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CPU, true)
	for name, fn := range map[string]func(){
		"eval_without_state":     func() { BatchNorm(x, nil, nil, nil, nil, BatchNormOptions{}) },
		"partial_state":          func() { BatchNorm(x, nil, nil, l.RunningMean, nil, BatchNormOptions{Training: true}) },
		"wrong_affine_shape":     func() { BatchNorm(x, wrong, nil, nil, nil, BatchNormOptions{Training: true}) },
		"state_requires_grad":    func() { BatchNorm(x, nil, nil, l.Weight, l.RunningVar, BatchNormOptions{Training: true}) },
		"state_alias":            func() { BatchNorm(x, nil, nil, l.RunningMean, l.RunningMean, BatchNormOptions{Training: true}) },
		"single_training_sample": func() { BatchNorm(single, nil, nil, nil, nil, BatchNormOptions{Training: true}) },
		"nan_epsilon":            func() { BatchNorm(x, nil, nil, nil, nil, BatchNormOptions{Training: true, Eps: float32(math.NaN())}) },
		"negative_epsilon":       func() { BatchNorm(x, nil, nil, nil, nil, BatchNormOptions{Training: true, Eps: -1}) },
		"invalid_momentum":       func() { BatchNorm(x, nil, nil, nil, nil, BatchNormOptions{Training: true, Momentum: 2}) },
	} {
		t.Run(name, func(t *testing.T) {
			panicked := false
			func() { defer func() { panicked = recover() != nil }(); fn() }()
			if !panicked {
				t.Fatal("invalid BatchNorm input accepted")
			}
		})
	}
	if _, err := NewBatchNormLayer(0, BatchNormLayerOptions{}, tensor.CPU); err == nil {
		t.Fatal("invalid constructor accepted")
	}
	context := NewExecutionContext()
	if err := context.Bind(l.RunningMean); err != nil {
		t.Fatal(err)
	}
	if err := context.Bind(l.RunningVar); err != nil {
		t.Fatal(err)
	}
	// A context supplied only by state still controls recording and identity.
	context.NoGrad(func() {
		y := BatchNorm(x, nil, nil, l.RunningMean, l.RunningVar, BatchNormOptions{})
		defer y.ReleaseGraph()
		if y.RequiresGrad || y.ExecutionContext() != context {
			t.Fatal("buffer-only context was ignored")
		}
	})
}

func TestBatchNormCUDAReplayAndMemory(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	for _, cumulative := range []bool{false, true} {
		name := "momentum"
		if cumulative {
			name = "cumulative"
		}
		t.Run(name, func(t *testing.T) {
			pool2dContext(t, tensor.CUDA)
			options := BatchNormLayerOptions{CumulativeMomentum: cumulative}
			layer, err := NewBatchNormLayer(3, options, tensor.CUDA)
			if err != nil {
				t.Fatal(err)
			}
			defer layer.Close()
			eager, err := NewBatchNormLayer(3, options, tensor.CUDA)
			if err != nil {
				t.Fatal(err)
			}
			defer eager.Close()
			context := NewExecutionContext()
			if err := context.BindModule(layer.StateModule()); err != nil {
				t.Fatal(err)
			}
			if err := context.BindModule(eager.StateModule()); err != nil {
				t.Fatal(err)
			}
			x := executionTensor(t, context, policyValues(120, .3), []int{2, 3, 4, 5}, tensor.CUDA, true)
			up := executionTensor(t, context, policyValues(120, 1.1), x.Shape, tensor.CUDA, false)
			step := func(l *BatchNormLayer) error {
				x.ZeroGrad()
				l.Weight.ZeroGrad()
				l.Bias.ZeroGrad()
				loss := weightedSum(l.Forward(x), up)
				defer loss.ReleaseGraph()
				return loss.Backward()
			}
			if err := step(layer); err != nil {
				t.Fatal(err)
			}
			stream, err := cuda.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Destroy()
			if err := SetBLASStream(stream); err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := SetBLASStream(nil); err != nil {
					t.Error(err)
				}
			}()
			graph, err := cuda.Capture(stream, func() error { return step(layer) })
			if err != nil {
				t.Fatal(err)
			}
			defer graph.Close()
			for i := 0; i < 2; i++ {
				if err := graph.Launch(); err != nil {
					t.Fatal(err)
				}
			}
			if err := stream.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if err := SetBLASStream(nil); err != nil {
				t.Fatal(err)
			}
			wantState := layer.Module.StateDict()
			wantGrad := conv2dHost(t, x, true)
			for i := 0; i < 3; i++ {
				if err := step(eager); err != nil {
					t.Fatal(err)
				}
			}
			for name, values := range eager.Module.StateDict() {
				conv2dClose(t, "captured "+name, wantState[name], values, 2e-6, 2e-5)
			}
			conv2dClose(t, "captured VJP", wantGrad, conv2dHost(t, x, true), 2e-6, 2e-5)
			graph.Close()
			before := cuda.MemoryStats().LiveBytes
			for i := 0; i < 10; i++ {
				context.NoGrad(func() { y := layer.Forward(x); y.ReleaseGraph() })
			}
			if cuda.MemoryStats().LiveBytes != before {
				t.Fatal("BatchNorm NoGrad leaked saved statistics")
			}
		})
	}
}
