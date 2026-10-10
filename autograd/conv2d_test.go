package autograd

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type conv2dFixtureCase struct {
	Name                 string
	XShape               []int `json:"x_shape"`
	WShape               []int `json:"w_shape"`
	YShape               []int `json:"y_shape"`
	Options              Conv2dOptions
	X, W, B, Upstream, Y []float32
	DX, DW, DB           []float32
}

func conv2dFixtures(t *testing.T) []conv2dFixtureCase {
	t.Helper()
	b, err := os.ReadFile("../testdata/conv2d.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct{ Cases []conv2dFixtureCase }
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("empty Conv2d fixture")
	}
	return f.Cases
}

func TestConv2dMixedExecutionContexts(t *testing.T) {
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
			input := conv2dTestTensor(t, []float32{1}, []int{1, 1, 1, 1}, device, false)
			weight := executionTensor(t, NewExecutionContext(), []float32{1}, []int{1, 1, 1, 1}, device, true)
			bias := executionTensor(t, NewExecutionContext(), []float32{1}, []int{1}, device, true)
			before := cuda.MemoryStats().LiveBytes
			panicked := false
			func() { defer func() { panicked = recover() != nil }(); Conv2d(input, weight, bias, Conv2dOptions{}) }()
			if !panicked {
				t.Fatal("mixed graphs were accepted")
			}
			if device == tensor.CUDA && before != cuda.MemoryStats().LiveBytes {
				t.Fatal("mixed-graph rejection leaked device memory")
			}
		})
	}
}

func TestConv2dPyTorchParity(t *testing.T) {
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
			for _, f := range conv2dFixtures(t) {
				t.Run(f.Name, func(t *testing.T) {
					x := conv2dTestTensor(t, f.X, f.XShape, device, true)
					w := conv2dTestTensor(t, f.W, f.WShape, device, true)
					var b *Tensor
					if f.B != nil {
						b = conv2dTestTensor(t, f.B, []int{f.WShape[0]}, device, true)
					}
					EnableGPUProfile(device == tensor.CUDA)
					defer EnableGPUProfile(false)
					y := Conv2d(x, w, b, f.Options)
					if !slices.Equal(y.Shape, f.YShape) {
						t.Fatalf("shape %v want %v", y.Shape, f.YShape)
					}
					if device == tensor.CUDA && (y.Buffer() == nil || y.Data != nil) {
						t.Fatal("Conv2d output left the device")
					}
					conv2dClose(t, "output", conv2dHost(t, y, false), f.Y, 3e-5, 3e-5)
					up := conv2dTestTensor(t, f.Upstream, f.YShape, device, false)
					loss := Sum(Mul(y, up), 0, 1, 2, 3)
					defer loss.ReleaseGraph()
					if err := loss.Backward(); err != nil {
						t.Fatal(err)
					}
					conv2dClose(t, "input grad", conv2dHost(t, x, true), f.DX, 5e-5, 5e-5)
					conv2dClose(t, "weight grad", conv2dHost(t, w, true), f.DW, 1e-4, 5e-5)
					if b != nil {
						conv2dClose(t, "bias grad", conv2dHost(t, b, true), f.DB, 1e-4, 5e-5)
					}
					if device == tensor.CUDA {
						for _, name := range []string{"c2_columns", "c2_col2im", "conv2d_forward_sgemm", "conv2d_dw_sgemm", "conv2d_dx_sgemm"} {
							if GPUProfile()[name].Count == 0 {
								t.Fatalf("no device execution for %s", name)
							}
						}
					}
				})
			}
		})
	}
}

func TestConv2dFiniteDifferences(t *testing.T) {
	x := Must([]float32{.2, -.1, .3, .4, .1, .6, -.5, .4, .3, -.2, .1, .7}, []int{1, 2, 2, 3}, true)
	w := Must([]float32{.1, -.4, .2, .3, -.2, .5, .3, .1}, []int{4, 1, 1, 2}, true)
	b := Must([]float32{.1, -.2, .05, .3}, []int{4}, true)
	o := Conv2dOptions{Groups: 2, Padding: [2]int{0, 1}}
	y := Conv2d(x, w, b, o)
	up := make([]float32, y.Numel())
	for i := range up {
		up[i] = float32(math.Cos(float64(i)*.31) * .3)
	}
	loss := Sum(Mul(y, Must(up, y.Shape, false)), 0, 1, 2, 3)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	loss.ReleaseGraph()
	evaluate := func() float64 {
		var sum float64
		NoGrad(func() {
			v := Conv2d(x, w, b, o)
			defer v.ReleaseGraph()
			for i, a := range v.Data {
				sum += float64(a) * float64(up[i])
			}
		})
		return sum
	}
	for name, p := range map[string]*Tensor{"input": x, "weight": w, "bias": b} {
		for i, original := range p.Data {
			const eps float32 = .001
			p.Data[i] = original + eps
			plus := evaluate()
			p.Data[i] = original - eps
			minus := evaluate()
			p.Data[i] = original
			want := (plus - minus) / float64(2*eps)
			if diff := math.Abs(float64(p.Grad[i]) - want); diff > 2e-4 {
				t.Fatalf("%s[%d] analytic=%g numerical=%g", name, i, p.Grad[i], want)
			}
		}
	}
}

// Transpose physical H/W axes before constructing the view, so both the input
// and filter arrive non-contiguous while retaining fixture values logically.
func conv2dTransposedStorage(v []float32, shape []int) ([]float32, []int) {
	baseShape := slices.Clone(shape)
	baseShape[2], baseShape[3] = shape[3], shape[2]
	out := make([]float32, len(v))
	for a := 0; a < shape[0]*shape[1]; a++ {
		for y := 0; y < shape[2]; y++ {
			for x := 0; x < shape[3]; x++ {
				out[(a*shape[3]+x)*shape[2]+y] = v[(a*shape[2]+y)*shape[3]+x]
			}
		}
	}
	return out, baseShape
}

func TestConv2dViewsAndSharedGradients(t *testing.T) {
	f := conv2dFixtures(t)[1]
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
			xv, xs := conv2dTransposedStorage(f.X, f.XShape)
			wv, ws := conv2dTransposedStorage(f.W, f.WShape)
			xbase := conv2dTestTensor(t, xv, xs, device, true)
			wbase := conv2dTestTensor(t, wv, ws, device, true)
			x, w := Transpose(xbase, 2, 3), Transpose(wbase, 2, 3)
			if x.IsContiguous() || w.IsContiguous() {
				t.Fatal("test views are contiguous")
			}
			b := conv2dTestTensor(t, f.B, []int{f.WShape[0]}, device, true)
			up := conv2dTestTensor(t, f.Upstream, f.YShape, device, false)
			// Two branches share the same views/parameters. Their gradients must
			// add, including scatter through the view's contiguous materialization.
			y := Add(Conv2d(x, w, b, f.Options), Conv2d(x, w, b, f.Options))
			loss := Sum(Mul(y, up), 0, 1, 2, 3)
			defer loss.ReleaseGraph()
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			for _, p := range []struct {
				name  string
				value *Tensor
				grad  []float32
				shape []int
			}{
				{"input", xbase, f.DX, f.XShape}, {"weight", wbase, f.DW, f.WShape},
			} {
				want, _ := conv2dTransposedStorage(p.grad, p.shape)
				for i := range want {
					want[i] *= 2
				}
				conv2dClose(t, p.name, conv2dHost(t, p.value, true), want, 1e-4, 5e-5)
			}
			want := slices.Clone(f.DB)
			for i := range want {
				want[i] *= 2
			}
			conv2dClose(t, "bias", conv2dHost(t, b, true), want, 1e-4, 5e-5)
		})
	}
}

func TestConv2dSelectiveGradients(t *testing.T) {
	f := conv2dFixtures(t)[0]
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
			for selected := 0; selected < 3; selected++ {
				x := conv2dTestTensor(t, f.X, f.XShape, device, selected == 0)
				w := conv2dTestTensor(t, f.W, f.WShape, device, selected == 1)
				b := conv2dTestTensor(t, f.B, []int{f.WShape[0]}, device, selected == 2)
				up := conv2dTestTensor(t, f.Upstream, f.YShape, device, false)
				loss := Sum(Mul(Conv2d(x, w, b, f.Options), up), 0, 1, 2, 3)
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				loss.ReleaseGraph()
				for i, p := range []*Tensor{x, w, b} {
					v := conv2dHost(t, p, true)
					if i == selected {
						conv2dClose(t, "selected grad", v, [][]float32{f.DX, f.DW, f.DB}[i], 1e-4, 5e-5)
					} else if len(v) != 0 {
						t.Fatal("gradient allocated for frozen tensor")
					}
				}
			}
			// NoGrad inference must release every temporary device allocation.
			x := conv2dTestTensor(t, f.X, f.XShape, device, true)
			w := conv2dTestTensor(t, f.W, f.WShape, device, true)
			before := cuda.MemoryStats().LiveBytes
			for i := 0; i < 5; i++ {
				NoGrad(func() {
					y := Conv2d(x, w, nil, f.Options)
					if y.RequiresGrad {
						t.Fatal("NoGrad recorded Conv2d")
					}
					y.ReleaseGraph()
				})
			}
			if device == tensor.CUDA && cuda.MemoryStats().LiveBytes != before {
				t.Fatal("inference live CUDA memory grew")
			}
		})
	}
}

func TestConv2dLayerState(t *testing.T) {
	for _, biased := range []bool{false, true} {
		l, err := NewConv2dLayer(4, 6, [2]int{2, 3}, Conv2dOptions{Groups: 2}, biased, tensor.CPU, rand.New(rand.NewSource(1)))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(l.Weight.Shape, []int{6, 2, 2, 3}) {
			t.Fatal(l.Weight.Shape)
		}
		if (l.Bias != nil) != biased {
			t.Fatal("bias choice not preserved")
		}
		root := Module{Children: []NamedModule{{"conv", l.StateModule()}}}
		root.Train(true)
		if !l.Module.Training {
			t.Fatal("child training propagation")
		}
		state := root.StateDict()
		if len(state) != 1+btoi(biased) || state["conv.weight"] == nil {
			t.Fatal("named state", state)
		}
		file := filepath.Join(t.TempDir(), "conv.safetensors")
		if err := root.SaveSafeTensors(file); err != nil {
			t.Fatal(err)
		}
		for i := range l.Weight.Data {
			l.Weight.Data[i] = 0
		}
		if err := root.LoadSafeTensors(file); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "restored weight", l.Weight.Data, state["conv.weight"], 0, 0)
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

func TestConv2dRejectsInvalidArguments(t *testing.T) {
	for _, test := range []struct {
		xs, ws []int
		o      Conv2dOptions
	}{
		{[]int{1, 2, 3}, []int{1, 2, 1, 1}, Conv2dOptions{}},
		{[]int{1, 2, 3, 3}, []int{3, 1, 1, 1}, Conv2dOptions{Groups: 2}},
		{[]int{1, 2, 3, 3}, []int{2, 1, 1, 1}, Conv2dOptions{}},
		{[]int{1, 2, 3, 3}, []int{2, 2, 5, 5}, Conv2dOptions{}},
		{[]int{1, 2, 3, 3}, []int{2, 2, 1, 1}, Conv2dOptions{Stride: [2]int{0, 1}}},
		{[]int{1, 2, 3, 3}, []int{2, 2, 1, 1}, Conv2dOptions{Padding: [2]int{-1, 0}}},
		{[]int{1, 2, 3, 3}, []int{2, 2, 1, 1}, Conv2dOptions{Dilation: [2]int{-1, 1}}},
		{[]int{1, 2, 3, 3}, []int{2, 2, 1, 1}, Conv2dOptions{Groups: -1}},
		{[]int{0, 2, 3, 3}, []int{2, 2, 1, 1}, Conv2dOptions{}},
		{[]int{1, 2, 1 << 30, 3}, []int{2, 2, 1, 1}, Conv2dOptions{}},
	} {
		if _, err := conv2dShape(test.xs, test.ws, test.o); err == nil {
			t.Fatalf("accepted invalid %+v", test)
		}
	}
	for _, bshape := range [][]int{{2, 1}, {1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("accepted malformed bias")
				}
			}()
			Conv2d(Must(make([]float32, 18), []int{1, 2, 3, 3}, false), Must(make([]float32, 4), []int{2, 2, 1, 1}, false), Must(make([]float32, numel(bshape)), bshape, false), Conv2dOptions{})
		}()
	}
	if _, err := NewConv2dLayer(4, 6, [2]int{0, 3}, Conv2dOptions{}, true, tensor.CPU, rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("accepted zero kernel")
	}
}

func TestConv2dScratchBound(t *testing.T) {
	s, err := conv2dShape([]int{1, 96, 2049, 67}, []int{96, 96, 3, 3}, Conv2dOptions{Padding: [2]int{1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	tile := s.tileColumns()
	if (s.depth()+s.outChannels())*tile*4 > 4<<20 {
		t.Fatal("tile exceeds scratch budget")
	}
	if tile >= s.oh*s.ow {
		t.Fatal("full im2col retained")
	}
}

func BenchmarkConv2dTrainingStep(b *testing.B) {
	x := Must(make([]float32, 2*8*32*32), []int{2, 8, 32, 32}, true)
	w := Must(make([]float32, 16*8*3*3), []int{16, 8, 3, 3}, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.ZeroGrad()
		w.ZeroGrad()
		loss := Mean(Conv2d(x, w, nil, Conv2dOptions{Padding: [2]int{1, 1}}), 0, 1, 2, 3)
		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}
		loss.ReleaseGraph()
	}
}
