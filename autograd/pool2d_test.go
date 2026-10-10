package autograd

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type pool2dFixture struct {
	Name, Kind         string
	XShape             []int `json:"x_shape"`
	YShape             []int `json:"y_shape"`
	X, Y, Upstream, DX []float32
	Options            struct {
		KernelSize, Stride, Padding, Dilation, OutputSize [2]int
		CeilMode, ExcludePad                              bool
		DivisorOverride                                   int
	}
}

func pool2dFixtures(t *testing.T) []pool2dFixture {
	t.Helper()
	data, err := os.ReadFile("../testdata/pool2d.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct{ Cases []pool2dFixture }
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures.Cases) == 0 {
		t.Fatal("empty pooling fixture")
	}
	return fixtures.Cases
}

func (f pool2dFixture) forward(x *Tensor) *Tensor {
	o := f.Options
	switch f.Kind {
	case "max":
		return MaxPool2d(x, MaxPool2dOptions{KernelSize: o.KernelSize, Stride: o.Stride, Padding: o.Padding, Dilation: o.Dilation, CeilMode: o.CeilMode})
	case "avg":
		return AvgPool2d(x, AvgPool2dOptions{KernelSize: o.KernelSize, Stride: o.Stride, Padding: o.Padding, CeilMode: o.CeilMode, ExcludePad: o.ExcludePad, DivisorOverride: o.DivisorOverride})
	case "adaptive":
		return AdaptiveAvgPool2d(x, o.OutputSize)
	default:
		panic("invalid pooling fixture kind")
	}
}

func pool2dContext(t *testing.T, device tensor.Device) {
	t.Helper()
	if device != tensor.CUDA {
		return
	}
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ctx.Close)
}

func pool2dDevices(t *testing.T, test func(*testing.T, tensor.Device)) {
	t.Helper()
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			pool2dContext(t, device)
			test(t, device)
		})
	}
}

func TestPool2dPyTorchParity(t *testing.T) {
	fixtures := pool2dFixtures(t)
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, f := range fixtures {
			t.Run(f.Name, func(t *testing.T) {
				pool2dContext(t, device)
				x := conv2dTestTensor(t, f.X, f.XShape, device, true)
				EnableGPUProfile(device == tensor.CUDA)
				defer EnableGPUProfile(false)
				y := f.forward(x)
				if !slices.Equal(y.Shape, f.YShape) {
					t.Fatalf("shape %v want %v", y.Shape, f.YShape)
				}
				if device == tensor.CUDA && (y.Buffer() == nil || y.Data != nil) {
					t.Fatal("pool output left CUDA")
				}
				conv2dClose(t, "pool output", conv2dHost(t, y, false), f.Y, 2e-6, 2e-5)
				up := conv2dTestTensor(t, f.Upstream, f.YShape, device, false)
				loss := weightedSum(y, up)
				defer loss.ReleaseGraph()
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "pool VJP", conv2dHost(t, x, true), f.DX, 2e-6, 2e-5)
				if device == tensor.CUDA {
					profile := GPUProfile()
					if profile["pool2d_f"].Count == 0 || profile["pool2d_b"].Count == 0 {
						t.Fatal("pool kernels did not execute")
					}
				}
			})
		}
	})
}

func TestPool2dFiniteDifferences(t *testing.T) {
	data := []float32{.2, -.4, .7, .1, .6, -.3, .4, .9, -.5, .3, -.1, .8}
	for name, forward := range map[string]func(*Tensor) *Tensor{
		"max": func(x *Tensor) *Tensor {
			return MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}, Stride: [2]int{1, 1}})
		},
		"avg": func(x *Tensor) *Tensor {
			return AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{3, 2}, Stride: [2]int{2, 2}, Padding: [2]int{1, 1}, CeilMode: true, ExcludePad: true})
		},
		"adaptive": func(x *Tensor) *Tensor { return AdaptiveAvgPool2d(x, [2]int{2, 3}) },
	} {
		t.Run(name, func(t *testing.T) {
			x := conv2dTestTensor(t, data, []int{1, 1, 3, 4}, tensor.CPU, true)
			y := forward(x)
			up := make([]float32, y.Numel())
			for i := range up {
				up[i] = float32(math.Cos(float64(i)*.41) * .3)
			}
			u := conv2dTestTensor(t, up, y.Shape, tensor.CPU, false)
			loss := weightedSum(y, u)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			want := conv2dHost(t, x, true)
			loss.ReleaseGraph()
			evaluate := func() float32 {
				var value float32
				NoGrad(func() {
					out := forward(x)
					for i, v := range out.Data {
						value += v * up[i]
					}
					out.ReleaseGraph()
				})
				return value
			}
			const epsilon = float32(.001)
			for i := range data {
				values := append([]float32(nil), data...)
				values[i] += epsilon
				if err := x.CopyFrom(values); err != nil {
					t.Fatal(err)
				}
				plus := evaluate()
				values[i] -= 2 * epsilon
				if err := x.CopyFrom(values); err != nil {
					t.Fatal(err)
				}
				minus := evaluate()
				got := (plus - minus) / (2 * epsilon)
				if math.Abs(float64(got-want[i])) > 1e-4 {
					t.Fatalf("gradient[%d] finite difference %g want %g", i, got, want[i])
				}
			}
		})
	}
}

func TestPool2dViewsBranchesAndRetainedGraph(t *testing.T) {
	fixtures := pool2dFixtures(t)
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, f := range []pool2dFixture{fixtures[1], fixtures[6], fixtures[10]} {
			t.Run(f.Name, func(t *testing.T) {
				pool2dContext(t, device)
				values, shape := conv2dTransposedStorage(f.X, f.XShape)
				context := NewExecutionContext()
				base := executionTensor(t, context, values, shape, device, true)
				x := Transpose(base, 2, 3)
				y := Add(f.forward(x), f.forward(x))
				if y.ExecutionContext() != context {
					t.Fatal("pooling lost execution context")
				}
				up := executionTensor(t, context, f.Upstream, f.YShape, device, false)
				loss := weightedSum(y, up)
				defer loss.ReleaseGraph()
				if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
					t.Fatal(err)
				}
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				want, _ := conv2dTransposedStorage(f.DX, f.XShape)
				for i := range want {
					want[i] *= 4
				}
				conv2dClose(t, "pool shared view gradient", conv2dHost(t, base, true), want, 5e-6, 5e-5)
			})
		}
	})
}

func TestMaxPool2dTiesNaNsAndEmptyWindows(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, c := range []struct {
			name     string
			data     []float32
			gradient []float32
			value    float32
		}{
			{"tie", []float32{2, 2, 1, 0}, []float32{1, 0, 0, 0}, 2},
			{"negative_infinity", []float32{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))}, []float32{1, 0, 0, 0}, float32(math.Inf(-1))},
			{"last_nan", []float32{float32(math.NaN()), 1, float32(math.NaN()), 2}, []float32{0, 0, 1, 0}, float32(math.NaN())},
		} {
			t.Run(c.name, func(t *testing.T) {
				pool2dContext(t, device)
				x := conv2dTestTensor(t, c.data, []int{1, 1, 2, 2}, device, true)
				y := MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}})
				got := conv2dHost(t, y, false)[0]
				if got != c.value && !(math.IsNaN(float64(got)) && math.IsNaN(float64(c.value))) {
					t.Fatalf("max %g want %g", got, c.value)
				}
				if err := y.Backward(); err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "max winner", conv2dHost(t, x, true), c.gradient, 0, 0)
				y.ReleaseGraph()
			})
		}
		// Dilation can leave a legal window containing only padding. It has
		// value -Inf and contributes no input gradient (no out-of-range index).
		x := conv2dTestTensor(t, []float32{1}, []int{1, 1, 1, 1}, device, true)
		y := MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}, Stride: [2]int{1, 1}, Padding: [2]int{1, 1}, Dilation: [2]int{2, 2}})
		if !math.IsInf(float64(conv2dHost(t, y, false)[0]), -1) {
			t.Fatal("empty max window did not return -Inf")
		}
		if err := y.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "empty max window gradient", conv2dHost(t, x, true), []float32{0}, 0, 0)
		y.ReleaseGraph()
	})
}

func TestPool2dSavedLifetimeAndNoGrad(t *testing.T) {
	fixtures := pool2dFixtures(t)
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		for _, f := range []pool2dFixture{fixtures[1], fixtures[5], fixtures[12]} {
			t.Run(f.Name, func(t *testing.T) {
				pool2dContext(t, device)
				context := NewExecutionContext()
				base := executionTensor(t, context, f.X, f.XShape, device, true)
				view := Reshape(base, base.Shape...)
				view.RetainGrad()
				defer view.Close()
				y := f.forward(view)
				up := executionTensor(t, context, f.Upstream, f.YShape, device, false)
				loss := weightedSum(y, up)
				defer loss.ReleaseGraph()
				base.Close()
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "closed base saved state", conv2dHost(t, view, true), f.DX, 2e-6, 2e-5)
				input := executionTensor(t, context, f.X, f.XShape, device, true)
				before := cuda.MemoryStats().LiveBytes
				for step := 0; step < 10; step++ {
					context.NoGrad(func() {
						out := f.forward(input)
						if out.RequiresGrad || out.backward != nil || out.backwardGPU != nil {
							t.Fatal("pool NoGrad recorded derivatives")
						}
						if f.Kind == "max" && device == tensor.CUDA && len(out.aux) != 0 {
							t.Fatal("max inference retained indices")
						}
						out.ReleaseGraph()
					})
				}
				if device == tensor.CUDA && cuda.MemoryStats().LiveBytes != before {
					t.Fatal("pool inference leaked CUDA memory")
				}
			})
		}
	})
}

func TestPool2dValidation(t *testing.T) {
	x := conv2dTestTensor(t, make([]float32, 16), []int{1, 1, 4, 4}, tensor.CPU, true)
	rankOne := conv2dTestTensor(t, []float32{1}, []int{1}, tensor.CPU, false)
	for name, fn := range map[string]func(){
		"missing_kernel":    func() { MaxPool2d(x, MaxPool2dOptions{}) },
		"partial_stride":    func() { AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{2, 2}, Stride: [2]int{0, 1}}) },
		"negative_padding":  func() { MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}, Padding: [2]int{-1, 0}}) },
		"oversized_padding": func() { AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{2, 2}, Padding: [2]int{2, 0}}) },
		"partial_dilation":  func() { MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}, Dilation: [2]int{1, 0}}) },
		"empty_output":      func() { MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{8, 8}}) },
		"negative_divisor":  func() { AvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{2, 2}, DivisorOverride: -1}) },
		"adaptive_zero":     func() { AdaptiveAvgPool2d(x, [2]int{0, 2}) },
		"adaptive_overflow": func() { AdaptiveAvgPool2d(x, [2]int{1 << 30, 2}) },
		"rank":              func() { AdaptiveAvgPool2d(rankOne, [2]int{1, 1}) },
	} {
		t.Run(name, func(t *testing.T) {
			panicked := false
			func() { defer func() { panicked = recover() != nil }(); fn() }()
			if !panicked {
				t.Fatal("invalid pool input accepted")
			}
		})
	}
}

func TestMaxPool2dLargeDilation(t *testing.T) {
	// A unit kernel is independent of dilation. Advancing its loop must not
	// overflow a native int on 32-bit CPU builds after visiting a valid sample.
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{1, 1, 2, 2}, device, true)
		y := MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{1, 1}, Dilation: [2]int{1<<31 - 1, 1<<31 - 1}})
		conv2dClose(t, "unit max kernel", conv2dHost(t, y, false), []float32{1, 2, 3, 4}, 0, 0)
		loss := Sum(y, 0, 1, 2, 3)
		defer loss.ReleaseGraph()
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "unit max gradient", conv2dHost(t, x, true), []float32{1, 1, 1, 1}, 0, 0)
	})
}

func TestPool2dSequentialAndCapture(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		context := NewExecutionContext()
		x := executionTensor(t, context, policyValues(84, .2), []int{1, 2, 6, 7}, device, true)
		layers := Sequential{
			MaxPool2dLayer{Options: MaxPool2dOptions{KernelSize: [2]int{2, 2}, Stride: [2]int{1, 1}, Padding: [2]int{1, 1}}},
			AvgPool2dLayer{Options: AvgPool2dOptions{KernelSize: [2]int{2, 2}, CeilMode: true}},
			AdaptiveAvgPool2dLayer{OutputSize: [2]int{2, 3}},
		}
		step := func() error {
			x.ZeroGrad()
			y := layers.Forward(x)
			if !slices.Equal(y.Shape, []int{1, 2, 2, 3}) {
				t.Fatalf("sequential shape %v", y.Shape)
			}
			loss := Sum(y, 0, 1, 2, 3)
			defer loss.ReleaseGraph()
			return loss.Backward()
		}
		if err := step(); err != nil {
			t.Fatal(err)
		}
		want := conv2dHost(t, x, true)
		if device != tensor.CUDA {
			return
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
		graph, err := cuda.Capture(stream, step)
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
		conv2dClose(t, "pool CUDA replay gradient", conv2dHost(t, x, true), want, 2e-6, 2e-5)
	})
}
