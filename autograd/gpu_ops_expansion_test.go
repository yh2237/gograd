package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/directml"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	"github.com/yh2237/gograd/vulkan"
)

func assertCloseWithNaN(t *testing.T, label string, got, want []float32, atol, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", label, len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(want[i])) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("%s[%d]: got %v, want NaN", label, i, got[i])
			}
			continue
		}
		if math.IsInf(float64(want[i]), 0) {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]: got %v, want %v", label, i, got[i], want[i])
			}
			continue
		}
		diff := math.Abs(float64(got[i] - want[i]))
		limit := atol + rtol*math.Abs(float64(want[i]))
		if diff > limit || math.IsNaN(diff) {
			t.Fatalf("%s[%d]: got %v, want %v (diff=%v > %v)", label, i, got[i], want[i], diff, limit)
		}
	}
}

func runCPUAndDevice(t *testing.T, device tensor.Device, label string, atol, rtol float64, build func(c *ExecutionContext, dev tensor.Device) (*Tensor, func())) {
	t.Helper()
	cpuCtx := NewExecutionContext()
	var cpuOut *Tensor
	var cpuClean func()
	cpuCtx.NoGrad(func() {
		cpuOut, cpuClean = build(cpuCtx, tensor.CPU)
	})
	want, err := cpuOut.ToHost()
	if err != nil {
		t.Fatalf("%s CPU ToHost: %v", label, err)
	}
	cpuOut.ReleaseGraph()
	cpuClean()

	devCtx := NewExecutionContext()
	var devOut *Tensor
	var devClean func()
	devCtx.NoGrad(func() {
		devOut, devClean = build(devCtx, device)
	})
	got, err := devOut.ToHost()
	if err != nil {
		t.Fatalf("%s %s ToHost: %v", label, device, err)
	}
	devOut.ReleaseGraph()
	devClean()
	assertCloseWithNaN(t, label+" ("+string(device)+")", got, want, atol, rtol)
}

func makeData(n int, scale, shift float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*17+5)%23-11)*scale + shift
	}
	return out
}

func availableInferenceDevices(t *testing.T) []tensor.Device {
	t.Helper()
	var devices []tensor.Device
	if directml.Available() {
		devices = append(devices, tensor.DirectML)
	}
	if vulkan.Available() {
		devices = append(devices, tensor.Vulkan)
	}
	if len(devices) == 0 {
		t.Skip("no DirectML or Vulkan device available")
	}
	return devices
}

func inferenceLiveBytes(dev tensor.Device) uint64 {
	if dev == tensor.DirectML {
		return directml.MemoryStats().LiveBytes
	}
	return vulkan.MemoryStats().LiveBytes
}

func TestInferenceOpsExpansion(t *testing.T) {
	aVals := makeData(24, 0.15, 0.05)
	bVals := makeData(24, 0.12, 1.8)
	posVals := makeData(24, 0.08, 2.2)
	biasVals := []float32{0.5, -1.25, 2.0, -0.75}

	for _, dev := range availableInferenceDevices(t) {
		t.Run(string(dev), func(t *testing.T) {
			before := inferenceLiveBytes(dev)

			for _, tc := range []struct {
				name string
				fn   func(a, b *Tensor) *Tensor
			}{
				{"Sub", Sub},
				{"Div", Div},
				{"Max", Max},
				{"Min", Min},
			} {
				runCPUAndDevice(t, dev, tc.name+"/same", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
					a, err := c.New(aVals, []int{2, 3, 4}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					b, err := c.New(bVals, []int{2, 3, 4}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					return tc.fn(a, b), func() { a.Close(); b.Close() }
				})
				runCPUAndDevice(t, dev, tc.name+"/broadcast_view", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
					a, err := c.New(aVals, []int{2, 4, 3}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					b, err := c.New(biasVals, []int{1, 1, 4}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					return tc.fn(Transpose(a, 1, 2), b), func() { a.Close(); b.Close() }
				})
			}

			for _, tc := range []struct {
				name string
				vals []float32
				fn   func(*Tensor) *Tensor
			}{
				{"AddScalar", aVals, func(x *Tensor) *Tensor { return AddScalar(x, 1.75) }},
				{"SubScalar", aVals, func(x *Tensor) *Tensor { return SubScalar(x, 0.625) }},
				{"DivScalar", aVals, func(x *Tensor) *Tensor { return DivScalar(x, 3.5) }},
				{"PowScalar", posVals, func(x *Tensor) *Tensor { return PowScalar(x, 1.5) }},
				{"Exp", aVals, Exp},
				{"Log", posVals, Log},
				{"Abs", aVals, Abs},
				{"Tanh", aVals, Tanh},
				{"ReLU", aVals, ReLU},
				{"Sin", aVals, Sin},
				{"Cos", aVals, Cos},
				{"Sqrt", posVals, Sqrt},
				{"Sigmoid", aVals, Sigmoid},
				{"Softplus", aVals, Softplus},
				{"Erf", aVals, Erf},
				{"Clamp", aVals, func(x *Tensor) *Tensor { return Clamp(x, -0.5, 0.75) }},
			} {
				runCPUAndDevice(t, dev, tc.name, 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
					x, err := c.New(tc.vals, []int{4, 6}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					return tc.fn(Transpose(x, 0, 1)), func() { x.Close() }
				})
			}

			for _, axis := range []int{0, 1, -1} {
				runCPUAndDevice(t, dev, "LogSoftmax", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
					x, err := c.New(aVals, []int{2, 3, 4}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					return LogSoftmax(x, axis), func() { x.Close() }
				})
			}
			{
				c := NewExecutionContext()
				emptyAxis, err := c.Zeros([]int{2, 0}, dev, false)
				if err != nil {
					t.Fatal(err)
				}
				assertGPUErrorPanic(t, gpu.ErrInvalidArgument, func() { c.NoGrad(func() { LogSoftmax(emptyAxis, -1) }) })
				emptyAxis.Close()

				emptyOuter, err := c.Zeros([]int{0, 2}, dev, false)
				if err != nil {
					t.Fatal(err)
				}
				var out *Tensor
				c.NoGrad(func() { out = LogSoftmax(emptyOuter, -1) })
				out.ReleaseGraph()
				emptyOuter.Close()
			}

			for _, tc := range []struct {
				name  string
				shape []int
				axes  []int
				mean  bool
			}{
				{"Sum/last", []int{2, 3, 4}, []int{-1}, false},
				{"Mean/mid", []int{2, 3, 4}, []int{1}, true},
				{"Sum/multi", []int{2, 3, 4, 5}, []int{0, 2}, false},
				{"Mean/all", []int{2, 3, 4}, []int{0, 1, 2}, true},
				{"Sum/rank6_alt", []int{2, 3, 2, 2, 2, 3}, []int{1, 3, 5}, false},
				{"Mean/rank6_alt", []int{2, 3, 2, 2, 2, 3}, []int{0, 2, 4}, true},
				{"Sum/no_axis", []int{2, 3}, nil, false},
				{"Mean/size1_axis", []int{2, 1, 3}, []int{1}, true},
				{"Sum/empty_axis", []int{0, 3}, []int{0}, false},
				{"Mean/empty_axis", []int{0, 3}, []int{0}, true},
				{"Sum/empty_out", []int{0, 3}, []int{1}, false},
			} {
				vals := makeData(numel(tc.shape), 0.1, 0.2)
				runCPUAndDevice(t, dev, tc.name, 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
					x, err := c.New(vals, tc.shape, d, false)
					if err != nil {
						t.Fatal(err)
					}
					if tc.mean {
						return Mean(x, tc.axes...), func() { x.Close() }
					}
					return Sum(x, tc.axes...), func() { x.Close() }
				})
			}

			if after := inferenceLiveBytes(dev); after != before {
				t.Fatalf("%s elementwise/reduce leaked: %d -> %d", dev, before, after)
			}
		})
	}
}

func TestInferenceVisionAndBatchNorm(t *testing.T) {
	xVals := makeData(2*4*7*8, 0.1, -0.05)
	wVals := makeData(6*2*3*3, 0.1, 0.02)
	bVals := makeData(6, 0.2, 0.1)
	bnMean := []float32{0.2, -0.5, 0.1, 0.7}
	bnVar := []float32{0.8, 1.5, 0.4, 2.2}
	bnWeight := []float32{1.2, 0.7, -0.9, 1.5}
	bnBias := []float32{0.1, -0.3, 0.4, -0.2}

	for _, dev := range availableInferenceDevices(t) {
		t.Run(string(dev), func(t *testing.T) {
			before := inferenceLiveBytes(dev)

			runCPUAndDevice(t, dev, "Conv2d/groups_dilated_bias", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(xVals, []int{2, 4, 7, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				w, err := c.New(wVals, []int{6, 2, 3, 3}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				b, err := c.New(bVals, []int{6}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return Conv2d(x, w, b, Conv2dOptions{Stride: [2]int{1, 2}, Padding: [2]int{1, 1}, Dilation: [2]int{2, 1}, Groups: 2}), func() {
					x.Close()
					w.Close()
					b.Close()
				}
			})

			runCPUAndDevice(t, dev, "MaxPool2d/ceil", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(xVals, []int{2, 4, 7, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{3, 2}, Stride: [2]int{2, 2}, Padding: [2]int{1, 0}, CeilMode: true}), func() { x.Close() }
			})

			runCPUAndDevice(t, dev, "MaxPool2d/dilated", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(xVals, []int{2, 4, 7, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}, Stride: [2]int{1, 2}, Padding: [2]int{0, 0}, Dilation: [2]int{2, 2}}), func() { x.Close() }
			})

			runCPUAndDevice(t, dev, "AvgPool2d/include_pad", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(xVals, []int{2, 4, 7, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{3, 3}, Stride: [2]int{2, 2}, Padding: [2]int{1, 1}}), func() { x.Close() }
			})

			runCPUAndDevice(t, dev, "AvgPool2d/ceil_exclude_pad", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(xVals, []int{2, 4, 7, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{3, 3}, Stride: [2]int{2, 2}, Padding: [2]int{1, 0}, CeilMode: true, ExcludePad: true}), func() { x.Close() }
			})

			runCPUAndDevice(t, dev, "AdaptiveAvgPool2d/global", 1e-5, 1e-5, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(xVals, []int{2, 4, 7, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return AdaptiveAvgPool2d(x, [2]int{1, 1}), func() { x.Close() }
			})

			for _, tc := range []struct {
				name         string
				shape        []int
				channelsLast bool
				hasW, hasB   bool
			}{
				{"NCHW_affine", []int{2, 4, 3, 5}, false, true, true},
				{"NCHW_no_affine", []int{2, 4, 3, 5}, false, false, false},
				{"ChannelsLast_weight_only", []int{2, 5, 3, 4}, true, true, false},
				{"Rank2_bias_only", []int{6, 4}, false, false, true},
			} {
				vals := makeData(numel(tc.shape), 0.2, 0.1)
				runCPUAndDevice(t, dev, "BatchNorm/"+tc.name, 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
					x, err := c.New(vals, tc.shape, d, false)
					if err != nil {
						t.Fatal(err)
					}
					m, err := c.New(bnMean, []int{4}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					v, err := c.New(bnVar, []int{4}, d, false)
					if err != nil {
						t.Fatal(err)
					}
					var w, b *Tensor
					if tc.hasW {
						w, err = c.New(bnWeight, []int{4}, d, false)
						if err != nil {
							t.Fatal(err)
						}
					}
					if tc.hasB {
						b, err = c.New(bnBias, []int{4}, d, false)
						if err != nil {
							t.Fatal(err)
						}
					}
					return BatchNorm(x, w, b, m, v, BatchNormOptions{ChannelsLast: tc.channelsLast, Eps: 1e-5}), func() {
						x.Close()
						m.Close()
						v.Close()
						if w != nil {
							w.Close()
						}
						if b != nil {
							b.Close()
						}
					}
				})
			}

			c := NewExecutionContext()
			x, err := c.New(xVals, []int{2, 4, 7, 8}, dev, false)
			if err != nil {
				t.Fatal(err)
			}
			assertGPUErrorPanic(t, gpu.ErrUnsupported, func() {
				c.NoGrad(func() { AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{2, 2}, DivisorOverride: 3}) })
			})
			assertGPUErrorPanic(t, gpu.ErrUnsupported, func() {
				c.NoGrad(func() {
					AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{3, 3}, Stride: [2]int{2, 2}, Padding: [2]int{0, 0}, CeilMode: true})
				})
			})
			assertGPUErrorPanic(t, gpu.ErrUnsupported, func() {
				c.NoGrad(func() { AdaptiveAvgPool2d(x, [2]int{3, 3}) })
			})
			x.Close()

			runCPUAndDevice(t, dev, "MiniCNN_Stack", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				rng := rand.New(rand.NewSource(42))
				conv1, err := NewConv2dLayer(3, 8, [2]int{3, 3}, Conv2dOptions{Padding: [2]int{1, 1}}, true, d, rng)
				if err != nil {
					t.Fatal(err)
				}
				bn1, err := NewBatchNormLayer(8, BatchNormLayerOptions{}, d)
				if err != nil {
					t.Fatal(err)
				}
				bn1.Train(false)
				conv2, err := NewConv2dLayer(8, 12, [2]int{3, 3}, Conv2dOptions{Padding: [2]int{1, 1}}, false, d, rng)
				if err != nil {
					t.Fatal(err)
				}
				head, err := NewLinearLayer(12, 5, d, rng)
				if err != nil {
					t.Fatal(err)
				}
				for _, m := range []*Module{conv1.StateModule(), bn1.StateModule(), conv2.StateModule(), head.StateModule()} {
					if err := c.BindModule(m); err != nil {
						t.Fatal(err)
					}
				}
				inVals := makeData(2*3*8*8, 0.1, 0.05)
				in, err := c.New(inVals, []int{2, 3, 8, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				h := ReLU(bn1.Forward(conv1.Forward(in)))
				h = MaxPool2dLayer{Options: MaxPool2dOptions{KernelSize: [2]int{2, 2}}}.Forward(h)
				h = ReLU(conv2.Forward(h))
				h = AdaptiveAvgPool2dLayer{OutputSize: [2]int{1, 1}}.Forward(h)
				h = Flatten(h, 1, -1)
				out := LogSoftmax(head.Forward(h), -1)
				return out, func() {
					in.Close()
					conv1.Weight.Close()
					conv1.Bias.Close()
					bn1.Close()
					conv2.Weight.Close()
					head.Weight.Close()
					head.Bias.Close()
				}
			})

			if after := inferenceLiveBytes(dev); after != before {
				t.Fatalf("%s vision/BN leaked: %d -> %d", dev, before, after)
			}
		})
	}
}

func TestInferenceModernTransformerBlocks(t *testing.T) {
	ropeVals := makeData(2*3*6*8, 0.1, 0.05)
	gegluVals := makeData(2*5*16, 0.1, -0.05)
	qVals := makeData(2*2*5*8, 0.1, 0.02)
	kVals := makeData(2*2*5*8, 0.1, -0.03)
	vVals := makeData(2*2*5*8, 0.1, 0.04)
	valid := []bool{true, true, true, true, false, true, true, true, false, false}
	rmsX := makeData(2*5*8, 0.15, 0.1)
	rmsW := makeData(8, 0.1, 1.0)
	swiX := makeData(2*4*8, 0.1, 0.05)
	w1Vals := makeData(16*8, 0.05, 0.01)
	w2Vals := makeData(8*16, 0.05, -0.01)
	w3Vals := makeData(16*8, 0.05, 0.02)

	for _, dev := range availableInferenceDevices(t) {
		t.Run(string(dev), func(t *testing.T) {
			before := inferenceLiveBytes(dev)

			runCPUAndDevice(t, dev, "RotaryHalf", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(ropeVals, []int{2, 3, 6, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return RotaryHalf(x, 10000), func() { x.Close() }
			})

			runCPUAndDevice(t, dev, "GeGLU", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(gegluVals, []int{2, 5, 16}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return GeGLU(x), func() { x.Close() }
			})

			runCPUAndDevice(t, dev, "KeyPaddingAttention/windowed", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				q, err := c.New(qVals, []int{2, 2, 5, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				k, err := c.New(kVals, []int{2, 2, 5, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				v, err := c.New(vVals, []int{2, 2, 5, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return KeyPaddingAttention(q, k, v, valid, 2), func() { q.Close(); k.Close(); v.Close() }
			})

			runCPUAndDevice(t, dev, "RMSNormLast", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(rmsX, []int{2, 5, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				w, err := c.New(rmsW, []int{8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return RMSNormLast(x, w, 1e-5), func() { x.Close(); w.Close() }
			})

			runCPUAndDevice(t, dev, "SwiGLUProjection", 1e-4, 1e-4, func(c *ExecutionContext, d tensor.Device) (*Tensor, func()) {
				x, err := c.New(swiX, []int{2, 4, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				w1, err := c.New(w1Vals, []int{16, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				w2, err := c.New(w2Vals, []int{8, 16}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				w3, err := c.New(w3Vals, []int{16, 8}, d, false)
				if err != nil {
					t.Fatal(err)
				}
				return SwiGLUProjection(x, w1, w2, w3), func() { x.Close(); w1.Close(); w2.Close(); w3.Close() }
			})

			c := NewExecutionContext()
			x, err := c.New(makeData(16, 0.1, 0.2), []int{2, 8}, dev, false)
			if err != nil {
				t.Fatal(err)
			}
			badW, err := c.New(makeData(16, 0.1, 0), []int{4, 4}, dev, false)
			if err != nil {
				t.Fatal(err)
			}
			assertGPUErrorPanic(t, gpu.ErrInvalidArgument, func() { c.NoGrad(func() { matMulTransB(x, badW) }) })
			x.Close()
			badW.Close()

			if after := inferenceLiveBytes(dev); after != before {
				t.Fatalf("%s leaked: %d -> %d", dev, before, after)
			}
		})
	}
}
