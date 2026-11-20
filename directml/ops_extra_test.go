package directml

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func checkExtra(t *testing.T, b *Backend, out *gpu.Buffer, err error, want []float32) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.Download(out, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range want {
		g := got[i]
		if math.IsNaN(float64(v)) && math.IsNaN(float64(g)) {
			continue
		}
		if g == v && (v != 0 || math.Signbit(float64(g)) == math.Signbit(float64(v))) {
			continue
		}
		if v != 0 && !math.IsInf(float64(v), 0) && math.Abs(float64(g-v)) <= 1e-4*(1+math.Abs(float64(v))) {
			continue
		}
		t.Fatalf("[%d] = %g, want %g", i, g, v)
	}
}

func TestExtraPointwise(t *testing.T) {
	b := testBackend(t)
	data := []float32{-2, -.5, 1, 4}
	right := []float32{2, 3, -2, 2}
	x, y, out := upload(t, b, data), upload(t, b, right), alloc(t, b, 4)
	binary := []func(float32, float32) float32{
		func(x, y float32) float32 { return x - y }, func(x, y float32) float32 { return x / y },
		func(x, y float32) float32 { return max(x, y) }, func(x, y float32) float32 { return min(x, y) },
	}
	for i, ref := range binary {
		t.Run(fmt.Sprintf("binary/%d", i), func(t *testing.T) {
			want := make([]float32, 4)
			for j, v := range data {
				want[j] = ref(v, right[j])
			}
			checkTransformerAfter(t, b, out, b.BinaryOp(gpu.BinaryOp(i), out, x, y, 4), want)
			a := gpu.View{Buffer: x, Shape: []int{2, 2}, Strides: []int{2, 1}}
			r := gpu.View{Buffer: y, Shape: []int{2, 2}, Strides: []int{0, 1}}
			for j, v := range data {
				want[j] = ref(v, right[j%2])
			}
			checkTransformerAfter(t, b, out, b.BinaryOpViews(gpu.BinaryOp(i), out, a, r), want)
		})
	}
	refs := []func(float64) float64{math.Exp, math.Log, math.Abs, math.Tanh,
		func(x float64) float64 { return max(0, x) }, math.Sin, math.Cos, math.Sqrt,
		func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }, func(x float64) float64 { return math.Log1p(math.Exp(x)) }, math.Erf}
	positive := []float32{.125, .5, 1, 4}
	p := upload(t, b, positive)
	info, err := b.DeviceInfo()
	if err != nil {
		t.Fatal(err)
	}
	for i, ref := range refs {
		t.Run(fmt.Sprintf("unary/%d", i), func(t *testing.T) {
			arg, values := x, data
			if i == int(gpu.UnaryLog) || i == int(gpu.UnarySqrt) {
				arg, values = p, positive
			}
			if i == int(gpu.UnaryErf) && info.FeatureLevel < 0x2000 {
				if err := b.UnaryOp(gpu.UnaryErf, out, arg, 4); !errors.Is(err, gpu.ErrUnsupported) {
					t.Fatal(err)
				}
				return
			}
			want := make([]float32, 4)
			for j, v := range values {
				want[j] = float32(ref(float64(v)))
			}
			checkTransformerAfter(t, b, out, b.UnaryOp(gpu.UnaryOp(i), out, arg, 4), want)
		})
	}
	for i, ref := range []func(float32) float32{
		func(x float32) float32 { return x + 2 }, func(x float32) float32 { return x - 2 },
		func(x float32) float32 { return x / 2 }, func(x float32) float32 { return x * x },
	} {
		want := make([]float32, 4)
		for j, v := range data {
			want[j] = ref(v)
		}
		checkTransformerAfter(t, b, out, b.ScalarOp(gpu.ScalarOp(i), out, x, 4, 2), want)
	}
	edge := []float32{-1, 0, 1, float32(math.Inf(1))}
	ex := upload(t, b, edge)
	for _, v := range []float32{0, math.Float32frombits(0x80000000), 2, -2, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		t.Run(fmt.Sprintf("divide/%08x", math.Float32bits(v)), func(t *testing.T) {
			want := make([]float32, 4)
			for i, x := range edge {
				want[i] = x / v
			}
			checkExtra(t, b, out, b.ScalarOp(gpu.ScalarDiv, out, ex, 4, v), want)
		})
	}
	checkTransformerAfter(t, b, out, b.Clamp(out, x, 4, -1, 2), []float32{-1, -.5, 1, 2})
	want := make([]float32, 4)
	for i := 0; i < 4; i += 2 {
		m := max(data[i], data[i+1])
		l := float64(m) + math.Log(math.Exp(float64(data[i]-m))+math.Exp(float64(data[i+1]-m)))
		want[i], want[i+1] = float32(float64(data[i])-l), float32(float64(data[i+1])-l)
	}
	checkTransformerAfter(t, b, out, b.LogSoftmax(out, x, 2, 2), want)
}

func checkTransformerAfter(t *testing.T, b *Backend, out *gpu.Buffer, err error, want []float32) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	checkTransformer(t, b, out, want)
}

func checkExtraBits(t *testing.T, b *Backend, out *gpu.Buffer, err error, want []float32) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.Download(out, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if g, w := math.Float32bits(v), math.Float32bits(want[i]); g != w {
			t.Errorf("[%d] = %08x, want %08x", i, g, w)
		}
	}
}

func TestPowScalarZeroAndOne(t *testing.T) {
	b := testBackend(t)
	negativeZero := math.Float32frombits(0x80000000)
	data := []float32{0, negativeZero, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc01234), math.SmallestNonzeroFloat32, -2}
	x, out := upload(t, b, data), alloc(t, b, len(data))
	ones := make([]float32, len(data))
	for i := range ones {
		ones[i] = 1
	}
	for _, exponent := range []float32{0, negativeZero, 1} {
		t.Run(fmt.Sprintf("%08x", math.Float32bits(exponent)), func(t *testing.T) {
			want := data
			if exponent == 0 {
				want = ones
			}
			checkExtraBits(t, b, out, b.ScalarOp(gpu.ScalarPow, out, x, len(data), exponent), want)
		})
	}
}

func TestPowScalarSpecialValues(t *testing.T) {
	b := testBackend(t)
	negativeZero := math.Float32frombits(0x80000000)
	positiveInf, negativeInf := float32(math.Inf(1)), float32(math.Inf(-1))
	nan := float32(math.NaN())
	const subnormal float32 = math.SmallestNonzeroFloat32
	data := []float32{
		negativeInf, -2, -1, -0x1p-70, -subnormal, negativeZero, 0, subnormal, 2.938737e-39, 0x1p-70, .25, .5,
		math.Nextafter32(1, 0), 1, math.Nextafter32(1, 2), 1.0000887, 2, 4.8740836e25, positiveInf, nan,
	}
	x, out := upload(t, b, data), alloc(t, b, len(data))
	for _, exponent := range []float32{
		.5, -.5, 2, -2, 3, -3, -1, 149, 150, 0x1p24, 1e6, -1e6, 1e9, -1e9,
		subnormal, -subnormal, positiveInf, negativeInf, nan,
	} {
		t.Run(fmt.Sprintf("%08x", math.Float32bits(exponent)), func(t *testing.T) {
			if err := b.ScalarOp(gpu.ScalarPow, out, x, len(data), exponent); err != nil {
				t.Fatal(err)
			}
			got, err := b.Download(out, len(data))
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				want := float32(math.Pow(float64(data[i]), float64(exponent)))
				if math.Float32bits(v) == math.Float32bits(want) || math.IsNaN(float64(v)) && math.IsNaN(float64(want)) {
					continue
				}
				if want != 0 && !math.IsInf(float64(want), 0) && math.Abs(float64(v)-float64(want)) <= 1e-5*math.Abs(float64(want)) {
					continue
				}
				t.Errorf("pow(%g, %g) = %g (%08x), want %g (%08x)", data[i], exponent, v, math.Float32bits(v), want, math.Float32bits(want))
			}
		})
	}
}

func TestClampFloat32SpecialValues(t *testing.T) {
	b := testBackend(t)
	negativeZero := math.Float32frombits(0x80000000)
	negativeInf, positiveInf := float32(math.Inf(-1)), float32(math.Inf(1))
	nan, negativeNaN := math.Float32frombits(0x7fc01234), math.Float32frombits(0xffc05678)
	const subnormal float32 = math.SmallestNonzeroFloat32
	special := []float32{nan, -2, negativeZero, 0, subnormal, -subnormal, positiveInf, negativeNaN, negativeInf, 2}
	for _, data := range [][]float32{{nan}, special} {
		x, out := upload(t, b, data), alloc(t, b, len(data))
		for _, bounds := range [][2]float32{
			{-1, 1}, {-2, -2}, {negativeInf, 1}, {-1, positiveInf}, {negativeInf, positiveInf},
			{-2, 0}, {-2, negativeZero}, {negativeZero, 2}, {0, 2}, {0, negativeZero}, {-subnormal, subnormal},
		} {
			lo, hi := bounds[0], bounds[1]
			t.Run(fmt.Sprintf("%d/%08x_%08x", len(data), math.Float32bits(lo), math.Float32bits(hi)), func(t *testing.T) {
				want := make([]float32, len(data))
				for i, v := range data {
					want[i] = v
					if !math.IsNaN(float64(v)) {
						want[i] = min(max(v, lo), hi)
					}
				}
				checkExtraBits(t, b, out, b.Clamp(out, x, len(data), lo, hi), want)
			})
		}
	}
}

func TestLogSoftmaxSpecialValues(t *testing.T) {
	b := testBackend(t)
	nan := float32(math.NaN())
	posInf, negInf := float32(math.Inf(1)), float32(math.Inf(-1))
	const cols = 4
	data := []float32{
		1, negInf, 1, negInf,
		1, nan, 2, negInf,
		1, posInf, 2, negInf,
		negInf, negInf, negInf, negInf,
		1e8, 1e8, 1e8, 1e8,
		0, -1000, 0, -1000,
	}
	half, quarter := -float32(math.Log(2)), -float32(math.Log(4))
	want := []float32{
		half, negInf, half, negInf,
		nan, nan, nan, nan,
		nan, nan, nan, nan,
		nan, nan, nan, nan,
		quarter, quarter, quarter, quarter,
		half, half - 1000, half, half - 1000,
	}
	x, out := upload(t, b, data), alloc(t, b, len(data))
	checkExtra(t, b, out, b.LogSoftmax(out, x, len(data)/cols, cols), want)
}

func TestExtraReduce(t *testing.T) {
	b := testBackend(t)
	for _, tc := range []struct{ shape, axes []int }{
		{nil, nil}, {[]int{3}, []int{0}}, {[]int{2, 3}, []int{-1}},
		{[]int{2, 3, 4}, []int{0, 2}}, {[]int{2, 3, 2, 2}, []int{1, 3}},
		{[]int{2, 3, 2, 2, 2}, []int{0, 2, 4}}, {[]int{2, 3, 2, 2, 3, 2}, []int{1, 3, 5}},
		{[]int{2, 3, 2, 2, 3, 2}, []int{0, 2, 4}}, {[]int{3, 1, 2}, []int{1}},
		{[]int{2, 0, 3}, []int{1}}, {[]int{2, 0, 3}, []int{0}},
	} {
		t.Run(fmt.Sprint(tc.shape, tc.axes), func(t *testing.T) {
			n, _ := gpu.Elements(tc.shape...)
			reduced := make([]bool, len(tc.shape))
			for _, a := range tc.axes {
				reduced[(a+len(tc.shape))%len(tc.shape)] = true
			}
			var kept []int
			count := 1
			for i, d := range tc.shape {
				if reduced[i] {
					count *= d
				} else {
					kept = append(kept, d)
				}
			}
			outN, _ := gpu.Elements(kept...)
			data, sums := make([]float32, n), make([]float32, outN)
			for i := range data {
				data[i] = float32(i%13 - 6)
				pos, idx, stride := i, 0, 1
				for a := len(tc.shape) - 1; a >= 0; a-- {
					d := tc.shape[a]
					if !reduced[a] {
						idx += (pos % d) * stride
						stride *= d
					}
					pos /= d
				}
				sums[idx] += data[i]
			}
			x, out := upload(t, b, data), alloc(t, b, outN)
			checkTransformerAfter(t, b, out, b.Reduce(gpu.ReduceSum, out, x, tc.shape, tc.axes), sums)
			for i := range sums {
				sums[i] /= float32(count)
			}
			checkExtra(t, b, out, b.Reduce(gpu.ReduceMean, out, x, tc.shape, tc.axes), sums)
		})
	}
}

func TestExtraConv2d(t *testing.T) {
	b := testBackend(t)
	info, err := b.DeviceInfo()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("adapter: %+v", info)
	for _, tc := range []struct {
		o    gpu.Conv2dOptions
		bias bool
		name string
	}{
		{gpu.Conv2dOptions{Batch: 2, InChannels: 4, Height: 4, Width: 5, OutChannels: 4, KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 1, PadH: 1, DilationH: 1, DilationW: 2, Groups: 2}, true, "grouped"},
		{gpu.Conv2dOptions{Batch: 1, InChannels: 1, Height: 3, Width: 3, OutChannels: 1, KernelH: 3, KernelW: 3, StrideH: 1, StrideW: 1, PadH: 1, PadW: 1, DilationH: 2, DilationW: 2, Groups: 1}, false, "single-position"},
		{gpu.Conv2dOptions{Batch: 1, InChannels: 1, Height: 3, Width: 4, OutChannels: 1, KernelH: 3, KernelW: 2, StrideH: 1, StrideW: 1, PadH: 1, DilationH: 2, DilationW: 1, Groups: 1}, false, "single-axis"},
		{gpu.Conv2dOptions{Batch: 1, InChannels: 1, Height: 1, Width: 3, OutChannels: 1, KernelH: 1, KernelW: 3, StrideH: 1, StrideW: 1, PadW: 1, DilationH: 1, DilationW: 2, Groups: 1}, false, "single-position-horizontal"},
		{gpu.Conv2dOptions{Batch: 1, InChannels: 1, Height: 3, Width: 1, OutChannels: 1, KernelH: 3, KernelW: 1, StrideH: 1, StrideW: 1, PadH: 1, DilationH: 2, DilationW: 1, Groups: 1}, false, "single-position-vertical"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := tc.o
			oh, ow, err := o.OutputShape()
			if err != nil {
				t.Fatal(err)
			}
			ic := o.InChannels / o.Groups
			x := make([]float32, o.Batch*o.InChannels*o.Height*o.Width)
			w := make([]float32, o.OutChannels*ic*o.KernelH*o.KernelW)
			bias := make([]float32, o.OutChannels)
			for i := range x {
				x[i] = float32(i%11+1) / 4
			}
			for i := range w {
				w[i] = float32(i%7+1) / 8
			}
			var v *gpu.Buffer
			if tc.bias {
				for i := range bias {
					bias[i] = float32(i) - 1
				}
				v = upload(t, b, bias)
			}
			want := make([]float32, o.Batch*o.OutChannels*oh*ow)
			for i := range want {
				n, c, h, k := i/(o.OutChannels*oh*ow), i/(oh*ow)%o.OutChannels, i/ow%oh, i%ow
				want[i] = bias[c]
				for j := 0; j < ic; j++ {
					xc := c/(o.OutChannels/o.Groups)*ic + j
					for kh := 0; kh < o.KernelH; kh++ {
						for kw := 0; kw < o.KernelW; kw++ {
							ih, iw := h*o.StrideH+kh*o.DilationH-o.PadH, k*o.StrideW+kw*o.DilationW-o.PadW
							if ih >= 0 && ih < o.Height && iw >= 0 && iw < o.Width {
								want[i] += x[((n*o.InChannels+xc)*o.Height+ih)*o.Width+iw] * w[((c*ic+j)*o.KernelH+kh)*o.KernelW+kw]
							}
						}
					}
				}
			}
			out := alloc(t, b, len(want))
			input, filter := upload(t, b, x), upload(t, b, w)
			err = b.Conv2d(out, input, filter, v, o)
			if info.VendorID == 0x1414 && info.DeviceID == 0x008c && oh == 1 && ow == 1 {
				if !errors.Is(err, gpu.ErrUnsupported) {
					t.Fatalf("got %v, want unsupported for WARP single-position dilation", err)
				}
				return
			}
			checkTransformerAfter(t, b, out, err, want)
		})
	}
}

func TestExtraPool2d(t *testing.T) {
	b := testBackend(t)
	info, err := b.DeviceInfo()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		avg, include, ceil bool
		dilation           int
	}{
		{false, false, false, 1}, {true, true, false, 1}, {true, false, false, 1}, {false, false, false, 2},
		{false, false, true, 1}, {true, false, true, 1}, {true, true, true, 1},
	} {
		t.Run(fmt.Sprint(tc), func(t *testing.T) {
			o := gpu.Pool2dOptions{Batch: 2, Channels: 2, Height: 3, Width: 3, KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2,
				PadH: 1, PadW: 1, EndPadH: 1, EndPadW: 1, DilationH: tc.dilation, DilationW: tc.dilation, OutH: 2, OutW: 2, IncludePad: tc.include}
			if tc.ceil {
				o.PadH, o.PadW = 0, 0
			}
			data := make([]float32, 36)
			for i := range data {
				data[i] = float32(i%13 - 13)
			}
			want := make([]float32, 16)
			for i := range want {
				base, h, w := i/4*9, i/2%2, i%2
				if !tc.avg {
					want[i] = float32(math.Inf(-1))
				}
				count := 0
				for kh := 0; kh < 2; kh++ {
					for kw := 0; kw < 2; kw++ {
						ih, iw := 2*h+kh*tc.dilation-o.PadH, 2*w+kw*tc.dilation-o.PadW
						if ih < 0 || ih >= 3 || iw < 0 || iw >= 3 {
							continue
						}
						v := data[base+ih*3+iw]
						count++
						if tc.avg {
							want[i] += v
						} else {
							want[i] = max(want[i], v)
						}
					}
				}
				if tc.avg {
					if tc.include {
						count = 4
					}
					want[i] /= float32(count)
				}
			}
			x, out := upload(t, b, data), alloc(t, b, len(want))
			if tc.avg {
				err = b.AvgPool2d(out, x, o)
			} else {
				err = b.MaxPool2d(out, x, o)
			}
			if tc.dilation == 2 && info.FeatureLevel < 0x2100 {
				if !errors.Is(err, gpu.ErrUnsupported) {
					t.Fatal(err)
				}
				return
			}
			checkTransformerAfter(t, b, out, err, want)
		})
	}
}

func TestExtraBatchNorm(t *testing.T) {
	b := testBackend(t)
	data := []float32{-1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3}
	means, vars := []float32{1, 2}, []float32{2, 4}
	weights, biases := []float32{2, 3}, []float32{4, 5}
	x, m, v := upload(t, b, data), upload(t, b, means), upload(t, b, vars)
	weight, bias := upload(t, b, weights), upload(t, b, biases)
	const eps = .01
	for _, w := range []*gpu.Buffer{nil, weight} {
		for _, a := range []*gpu.Buffer{nil, bias} {
			want := make([]float32, len(data))
			for i, x := range data {
				c := i / 3 % 2
				y := (x - means[c]) / float32(math.Sqrt(float64(vars[c]+eps)))
				if w != nil {
					y *= weights[c]
				}
				if a != nil {
					y += biases[c]
				}
				want[i] = y
			}
			out := alloc(t, b, len(data))
			checkTransformerAfter(t, b, out, b.BatchNormEval(out, x, w, a, m, v, 2, 2, 3, eps), want)
		}
	}
	nan := float32(math.NaN())
	mean := upload(t, b, []float32{float32(math.Inf(1)), nan})
	out := alloc(t, b, 2)
	checkExtra(t, b, out, b.BatchNormEval(out, upload(t, b, data[:2]), nil, nil, mean, v, 1, 2, 1, eps), []float32{float32(math.Inf(-1)), nan})
}

func TestExtraValidationDoesNotPoison(t *testing.T) {
	b := testBackend(t)
	x, y, empty, emptyOut := upload(t, b, []float32{1, 2, 3, 4}), alloc(t, b, 4), alloc(t, b, 0), alloc(t, b, 0)
	view := gpu.View{Buffer: x, Shape: []int{2, 2}, Strides: []int{2, 1}}
	pool := gpu.Pool2dOptions{Batch: 1, Channels: 1, Height: 2, Width: 2, KernelH: 1, KernelW: 1, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, OutH: 2, OutW: 2}
	conv := gpu.Conv2dOptions{Batch: 1, InChannels: 1, Height: 2, Width: 2, OutChannels: 1, KernelH: 1, KernelW: 1, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, Groups: 1}
	poolOverflow := gpu.Pool2dOptions{
		Batch: 1, Channels: 1, Height: 3, Width: 3,
		KernelH: 3, KernelW: 3, StrideH: math.MaxInt32, StrideW: 1,
		PadH: math.MaxInt32, EndPadH: math.MaxInt32, DilationH: 1, DilationW: 1,
		OutH: 3, OutW: 1,
	}
	bigIn, bigOut := upload(t, b, make([]float32, 9)), alloc(t, b, 3)
	for _, err := range []error{
		b.BinaryOp(gpu.BinarySub, x, x, x, 4), b.BinaryOp(gpu.BinaryDiv, y, x, x, 5),
		b.BinaryOpViews(gpu.BinaryMin, x, view, view), b.ScalarOp(gpu.ScalarAdd, x, x, 4, 1),
		b.ScalarOp(gpu.ScalarDiv, x, x, 4, 0), b.UnaryOp(gpu.UnaryExp, x, x, 4), b.Clamp(x, x, 4, 0, 1),
		b.ScalarOp(gpu.ScalarPow, x, x, 4, 0), b.ScalarOp(gpu.ScalarPow, x, x, 4, 1),
		b.LogSoftmax(x, x, 2, 2), b.LogSoftmax(y, x, 2, 0),
		b.Reduce(gpu.ReduceSum, x, x, []int{2, 2}, []int{0}), b.Reduce(gpu.ReduceMean, y, x, []int{2, 2}, []int{-1, 1}),
		b.Reduce(gpu.ReduceMean, y, x, []int{2, 2}, []int{2}), b.Conv2d(x, x, y, nil, conv), b.Conv2d(y, x, x, y, conv),
		b.MaxPool2d(x, x, pool), b.AvgPool2d(x, x, pool), b.MaxPool2d(y, x, gpu.Pool2dOptions{}),
		b.MaxPool2d(bigOut, bigIn, poolOverflow), b.AvgPool2d(bigOut, bigIn, poolOverflow),
		b.BatchNormEval(y, x, nil, nil, y, x, 1, 1, 4, .01), b.BatchNormEval(y, x, nil, nil, x, x, 1, 1, 4, 0),
	} {
		if !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Errorf("got %v, want invalid argument", err)
		}
	}
	nearMax := upload(t, b, []float32{1.020847e37})
	nearOut := alloc(t, b, 1)
	checkExtra(t, b, nearOut, b.ScalarOp(gpu.ScalarDiv, nearOut, nearMax, 1, 0.03), []float32{float32(1.020847e37) / float32(0.03)})
	pool.DilationH = 2
	for _, err := range []error{b.AvgPool2d(y, x, pool), b.Clamp(y, x, 4, 2, 1), b.Clamp(y, x, 4, 0, float32(math.NaN())), b.ScalarOp(gpu.ScalarDiv, y, x, 4, math.SmallestNonzeroFloat32)} {
		if !errors.Is(err, gpu.ErrUnsupported) {
			t.Errorf("got %v, want unsupported", err)
		}
	}
	for _, err := range []error{
		b.UnaryOp(gpu.UnaryLog, emptyOut, empty, 0), b.Clamp(emptyOut, empty, 0, 0, 1),
		b.ScalarOp(gpu.ScalarDiv, emptyOut, empty, 0, 0), b.BinaryOp(gpu.BinarySub, emptyOut, empty, empty, 0),
		b.ScalarOp(gpu.ScalarPow, emptyOut, empty, 0, 0), b.ScalarOp(gpu.ScalarPow, emptyOut, empty, 0, 1),
		b.LogSoftmax(emptyOut, empty, 0, 4),
	} {
		if err != nil {
			t.Error(err)
		}
	}
	checkTransformerAfter(t, b, y, b.Clamp(y, x, 4, 1, 3), []float32{1, 2, 3, 3})
}
