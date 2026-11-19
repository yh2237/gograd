package vulkan

import (
	"errors"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func assertClose(t *testing.T, label string, got, want []float32, atol, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", label, len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(want[i])) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("%s[%d] = %v, want NaN", label, i, got[i])
			}
			continue
		}
		if math.IsInf(float64(want[i]), 0) {
			if got[i] != want[i] {
				t.Fatalf("%s[%d] = %v, want %v", label, i, got[i], want[i])
			}
			continue
		}
		diff := math.Abs(float64(got[i] - want[i]))
		limit := atol + rtol*math.Abs(float64(want[i]))
		if diff > limit || math.IsNaN(diff) {
			t.Fatalf("%s[%d] = %v, want %v (diff=%v > %v)", label, i, got[i], want[i], diff, limit)
		}
	}
}

func TestVulkanExtraOps(t *testing.T) {
	if !Available() {
		t.Skipf("Vulkan unavailable: %v", Err())
	}
	b := NewBackend()
	defer func() {
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	aVals := []float32{-3.5, -1, 0, 0.25, 2.5, 11.75}
	bVals := []float32{1.25, -4, 0.5, 2, -0.5, 3.25}
	posVals := []float32{0.1, 0.5, 1, 2.5, 4, 9.25}
	aBuf, err := b.Upload(aVals)
	if err != nil {
		t.Fatal(err)
	}
	defer aBuf.Close()
	bBuf, err := b.Upload(bVals)
	if err != nil {
		t.Fatal(err)
	}
	defer bBuf.Close()
	posBuf, err := b.Upload(posVals)
	if err != nil {
		t.Fatal(err)
	}
	defer posBuf.Close()
	out, err := b.Alloc(len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	for _, tc := range []struct {
		name string
		op   gpu.BinaryOp
		fn   func(x, y float32) float32
	}{
		{"Sub", gpu.BinarySub, func(x, y float32) float32 { return x - y }},
		{"Div", gpu.BinaryDiv, func(x, y float32) float32 { return x / y }},
		{"Max", gpu.BinaryMax, func(x, y float32) float32 { return max(x, y) }},
		{"Min", gpu.BinaryMin, func(x, y float32) float32 { return min(x, y) }},
	} {
		want := make([]float32, len(aVals))
		for i := range aVals {
			want[i] = tc.fn(aVals[i], bVals[i])
		}
		if err := b.BinaryOp(tc.op, out, aBuf, bBuf, len(aVals)); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		got, err := b.Download(out, len(aVals))
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, tc.name, got, want, 1e-5, 1e-5)

		av := gpu.View{Buffer: aBuf, Shape: []int{2, 3}, Strides: []int{3, 1}}
		bv := gpu.View{Buffer: bBuf, Shape: []int{2, 3}, Strides: []int{0, 1}}
		if err := b.BinaryOpViews(tc.op, out, av, bv); err != nil {
			t.Fatalf("%s view: %v", tc.name, err)
		}
		got, err = b.Download(out, len(aVals))
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			want[i] = tc.fn(aVals[i], bVals[i%3])
		}
		assertClose(t, tc.name+" view", got, want, 1e-5, 1e-5)
	}

	for _, tc := range []struct {
		name string
		op   gpu.ScalarOp
		in   *gpu.Buffer
		vals []float32
		v    float32
		fn   func(float32) float32
	}{
		{"ScalarAdd", gpu.ScalarAdd, aBuf, aVals, 2.5, func(x float32) float32 { return x + 2.5 }},
		{"ScalarSub", gpu.ScalarSub, aBuf, aVals, 1.5, func(x float32) float32 { return x - 1.5 }},
		{"ScalarDiv", gpu.ScalarDiv, aBuf, aVals, 0.3, func(x float32) float32 { return x / 0.3 }},
		{"ScalarPow/frac", gpu.ScalarPow, posBuf, posVals, 1.5, func(x float32) float32 { return float32(math.Pow(float64(x), 1.5)) }},
		{"ScalarPow/neg_odd", gpu.ScalarPow, aBuf, aVals, 3, func(x float32) float32 { return x * x * x }},
		{"ScalarPow/zero", gpu.ScalarPow, aBuf, aVals, 0, func(float32) float32 { return 1 }},
	} {
		want := make([]float32, len(tc.vals))
		for i, x := range tc.vals {
			want[i] = tc.fn(x)
		}
		if err := b.ScalarOp(tc.op, out, tc.in, len(tc.vals), tc.v); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		got, err := b.Download(out, len(tc.vals))
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, tc.name, got, want, 1e-4, 1e-4)
	}

	nearMax, err := b.Upload([]float32{1.020847e37})
	if err != nil {
		t.Fatal(err)
	}
	defer nearMax.Close()
	nearOut, err := b.Alloc(1)
	if err != nil {
		t.Fatal(err)
	}
	defer nearOut.Close()
	if err := b.ScalarOp(gpu.ScalarDiv, nearOut, nearMax, 1, 0.03); err != nil {
		t.Fatal(err)
	}
	gotNear, err := b.Download(nearOut, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "ScalarDiv near MaxFloat32", gotNear, []float32{float32(1.020847e37) / float32(0.03)}, 0, 1e-5)

	for _, tc := range []struct {
		name string
		op   gpu.UnaryOp
		in   *gpu.Buffer
		vals []float32
		fn   func(float32) float32
	}{
		{"Exp", gpu.UnaryExp, aBuf, aVals, func(x float32) float32 { return float32(math.Exp(float64(x))) }},
		{"Log", gpu.UnaryLog, posBuf, posVals, func(x float32) float32 { return float32(math.Log(float64(x))) }},
		{"Abs", gpu.UnaryAbs, aBuf, aVals, func(x float32) float32 { return float32(math.Abs(float64(x))) }},
		{"Tanh", gpu.UnaryTanh, aBuf, aVals, func(x float32) float32 { return float32(math.Tanh(float64(x))) }},
		{"ReLU", gpu.UnaryReLU, aBuf, aVals, func(x float32) float32 {
			if x > 0 {
				return x
			}
			return 0
		}},
		{"Sin", gpu.UnarySin, aBuf, aVals, func(x float32) float32 { return float32(math.Sin(float64(x))) }},
		{"Cos", gpu.UnaryCos, aBuf, aVals, func(x float32) float32 { return float32(math.Cos(float64(x))) }},
		{"Sqrt", gpu.UnarySqrt, posBuf, posVals, func(x float32) float32 { return float32(math.Sqrt(float64(x))) }},
		{"Sigmoid", gpu.UnarySigmoid, aBuf, aVals, func(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }},
		{"Softplus", gpu.UnarySoftplus, aBuf, aVals, func(x float32) float32 {
			return max(x, 0) + float32(math.Log1p(math.Exp(-math.Abs(float64(x)))))
		}},
		{"Erf", gpu.UnaryErf, aBuf, aVals, func(x float32) float32 { return float32(math.Erf(float64(x))) }},
	} {
		want := make([]float32, len(tc.vals))
		for i, x := range tc.vals {
			want[i] = tc.fn(x)
		}
		if err := b.UnaryOp(tc.op, out, tc.in, len(tc.vals)); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		got, err := b.Download(out, len(tc.vals))
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, tc.name, got, want, 1e-4, 1e-4)
	}

	if err := b.Clamp(out, aBuf, len(aVals), -1, 2); err != nil {
		t.Fatal(err)
	}
	gotClamp, err := b.Download(out, len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	wantClamp := make([]float32, len(aVals))
	for i, x := range aVals {
		wantClamp[i] = min(max(x, -1), 2)
	}
	assertClose(t, "Clamp", gotClamp, wantClamp, 1e-6, 1e-6)

	if err := b.LogSoftmax(out, aBuf, 2, 3); err != nil {
		t.Fatal(err)
	}
	gotLS, err := b.Download(out, len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	wantLS := make([]float32, len(aVals))
	for r := 0; r < 2; r++ {
		mx := max(aVals[r*3], max(aVals[r*3+1], aVals[r*3+2]))
		var sum float64
		for c := 0; c < 3; c++ {
			sum += math.Exp(float64(aVals[r*3+c] - mx))
		}
		lse := float32(math.Log(sum))
		for c := 0; c < 3; c++ {
			wantLS[r*3+c] = aVals[r*3+c] - mx - lse
		}
	}
	assertClose(t, "LogSoftmax", gotLS, wantLS, 1e-5, 1e-5)

	maxVals, err := b.Upload([]float32{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32})
	if err != nil {
		t.Fatal(err)
	}
	defer maxVals.Close()
	redOut, err := b.Alloc(2)
	if err != nil {
		t.Fatal(err)
	}
	defer redOut.Close()
	if err := b.Reduce(gpu.ReduceMean, redOut, maxVals, []int{2, 2}, []int{1}); err != nil {
		t.Fatal(err)
	}
	gotRed, err := b.Download(redOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "ReduceMean MaxFloat32", gotRed, []float32{math.MaxFloat32, math.MaxFloat32}, 0, 1e-5)

	empty, err := b.Alloc(0)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	if err := b.Reduce(gpu.ReduceSum, redOut, empty, []int{0, 2}, []int{0}); err != nil {
		t.Fatal(err)
	}
	gotEmptySum, err := b.Download(redOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "ReduceSum empty", gotEmptySum, []float32{0, 0}, 0, 0)
	if err := b.Reduce(gpu.ReduceMean, redOut, empty, []int{0, 2}, []int{0}); err != nil {
		t.Fatal(err)
	}
	gotEmptyMean, err := b.Download(redOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "ReduceMean empty", gotEmptyMean, []float32{float32(math.NaN()), float32(math.NaN())}, 0, 0)

	// Validation errors must not poison the backend.
	poolOverflow := gpu.Pool2dOptions{
		Batch: 1, Channels: 1, Height: 3, Width: 3,
		KernelH: 3, KernelW: 3, StrideH: math.MaxInt32, StrideW: 1,
		PadH: math.MaxInt32, EndPadH: math.MaxInt32, DilationH: 1, DilationW: 1,
		OutH: 3, OutW: 1,
	}
	for _, err := range []error{
		b.BinaryOp(gpu.BinarySub, aBuf, aBuf, bBuf, len(aVals)),
		b.ScalarOp(gpu.ScalarAdd, aBuf, aBuf, len(aVals), 1),
		b.UnaryOp(gpu.UnaryExp, aBuf, aBuf, len(aVals)),
		b.LogSoftmax(out, aBuf, 2, 0),
		b.Reduce(gpu.ReduceSum, out, aBuf, []int{2, 3}, []int{2}),
		b.MaxPool2d(out, aBuf, poolOverflow),
		b.AvgPool2d(out, aBuf, poolOverflow),
		b.BatchNormEval(out, aBuf, nil, nil, aBuf, aBuf, 1, 2, 3, 0),
	} {
		if !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatalf("got %v, want ErrInvalidArgument", err)
		}
	}
	if err := b.AvgPool2d(out, aBuf, gpu.Pool2dOptions{
		Batch: 1, Channels: 1, Height: 2, Width: 3, KernelH: 1, KernelW: 1,
		StrideH: 1, StrideW: 1, DilationH: 2, DilationW: 1, OutH: 2, OutW: 3,
	}); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatalf("dilated AvgPool2d = %v, want ErrUnsupported", err)
	}

	if err := b.BinaryOp(gpu.BinarySub, out, aBuf, bBuf, len(aVals)); err != nil {
		t.Fatalf("backend poisoned after validation errors: %v", err)
	}

	// Regression checks for GLSL special values, signed zero, Softplus precision, and AvgPool2d divisor.
	negZero := float32(math.Copysign(0, -1))
	specialIn, err := b.Upload([]float32{negZero, 1, float32(math.Inf(-1)), -18, 0, -2})
	if err != nil {
		t.Fatal(err)
	}
	defer specialIn.Close()
	specOut, err := b.Alloc(6)
	if err != nil {
		t.Fatal(err)
	}
	defer specOut.Close()

	if err := b.ScalarOp(gpu.ScalarPow, specOut, specialIn, 6, 3); err != nil {
		t.Fatal(err)
	}
	gotPow3, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(gotPow3[0]) != math.Float32bits(negZero) {
		t.Fatalf("Pow(-0, 3) bits = %x, want -0", math.Float32bits(gotPow3[0]))
	}

	if err := b.ScalarOp(gpu.ScalarPow, specOut, specialIn, 6, -3); err != nil {
		t.Fatal(err)
	}
	gotPowNeg3, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(float64(gotPowNeg3[0]), -1) {
		t.Fatalf("Pow(-0, -3) = %v, want -Inf", gotPowNeg3[0])
	}

	if err := b.ScalarOp(gpu.ScalarPow, specOut, specialIn, 6, 0.5); err != nil {
		t.Fatal(err)
	}
	gotPowHalf, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(float64(gotPowHalf[2]), 1) {
		t.Fatalf("Pow(-Inf, 0.5) = %v, want +Inf", gotPowHalf[2])
	}

	if err := b.ScalarOp(gpu.ScalarPow, specOut, specialIn, 6, float32(math.NaN())); err != nil {
		t.Fatal(err)
	}
	gotPowNaN, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	if gotPowNaN[1] != 1 {
		t.Fatalf("Pow(1, NaN) = %v, want 1", gotPowNaN[1])
	}

	if err := b.UnaryOp(gpu.UnarySoftplus, specOut, specialIn, 6); err != nil {
		t.Fatal(err)
	}
	gotSP, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	wantSP18 := float32(math.Log1p(math.Exp(-18)))
	if math.Abs(float64(gotSP[3]-wantSP18)) > 1e-10 {
		t.Fatalf("Softplus(-18) = %g, want %g", gotSP[3], wantSP18)
	}

	if err := b.UnaryOp(gpu.UnaryLog, specOut, specialIn, 6); err != nil {
		t.Fatal(err)
	}
	gotLog, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(float64(gotLog[4]), -1) || !math.IsNaN(float64(gotLog[5])) {
		t.Fatalf("Log(0,-2) = %v,%v, want -Inf,NaN", gotLog[4], gotLog[5])
	}

	if err := b.UnaryOp(gpu.UnarySqrt, specOut, specialIn, 6); err != nil {
		t.Fatal(err)
	}
	gotSqrt, err := b.Download(specOut, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(float64(gotSqrt[5])) {
		t.Fatalf("Sqrt(-2) = %v, want NaN", gotSqrt[5])
	}

	pz, err := b.Upload([]float32{0, negZero})
	if err != nil {
		t.Fatal(err)
	}
	defer pz.Close()
	nz, err := b.Upload([]float32{negZero, 0})
	if err != nil {
		t.Fatal(err)
	}
	defer nz.Close()
	zOut, err := b.Alloc(2)
	if err != nil {
		t.Fatal(err)
	}
	defer zOut.Close()
	if err := b.BinaryOp(gpu.BinaryMax, zOut, pz, nz, 2); err != nil {
		t.Fatal(err)
	}
	gotMaxZ, err := b.Download(zOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(gotMaxZ[0]) != 0 || math.Float32bits(gotMaxZ[1]) != 0 {
		t.Fatalf("Max(±0,∓0) bits = %x,%x, want +0,+0", math.Float32bits(gotMaxZ[0]), math.Float32bits(gotMaxZ[1]))
	}
	if err := b.BinaryOp(gpu.BinaryMin, zOut, pz, nz, 2); err != nil {
		t.Fatal(err)
	}
	gotMinZ, err := b.Download(zOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(gotMinZ[0]) != 0x80000000 || math.Float32bits(gotMinZ[1]) != 0x80000000 {
		t.Fatalf("Min(±0,∓0) bits = %x,%x, want -0,-0", math.Float32bits(gotMinZ[0]), math.Float32bits(gotMinZ[1]))
	}

	if err := b.AvgPool2d(zOut, pz, gpu.Pool2dOptions{
		Batch: 1, Channels: 1, Height: 1, Width: 2, KernelH: 1, KernelW: 2,
		StrideH: 1, StrideW: 2, PadH: 0, PadW: 0, EndPadH: 0, EndPadW: 1,
		DilationH: 1, DilationW: 1, OutH: 1, OutW: 1, IncludePad: true,
	}); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatalf("AvgPool2d extended EndPad with IncludePad = %v, want ErrUnsupported", err)
	}
}
