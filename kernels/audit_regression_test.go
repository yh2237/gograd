package kernels

import (
	"math"
	"runtime"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

func auditCUDAKernels(t *testing.T) {
	t.Helper()
	if !cuda.Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
}

// referenceConvWeightGrad returns dw[o, c*kernel+k] for the centered tap
// offset that the forward im2col uses. x is channel-major [batch,channels,length]
// and dy is [batch,outChannels,length], the layout the fallback kernel reads.
func referenceConvWeightGrad(x, dy []float32, batch, channels, length, outChannels, kernel, dilation int) []float32 {
	dw := make([]float32, outChannels*channels*kernel)
	for o := 0; o < outChannels; o++ {
		for c := 0; c < channels; c++ {
			for k := 0; k < kernel; k++ {
				sum := float32(0)
				for b := 0; b < batch; b++ {
					for t := 0; t < length; t++ {
						source := t - (kernel/2)*dilation + k*dilation
						if source < 0 || source >= length {
							continue
						}
						sum += dy[(b*outChannels+o)*length+t] * x[(b*channels+c)*length+source]
					}
				}
				dw[(o*channels+c)*kernel+k] = sum
			}
		}
	}
	return dw
}

// TestAuditConvWeightGradTapAlignment checks that the standalone weight
// gradient fallback uses the same centered tap offset as the forward im2col.
// An uncentered offset silently mixes different input samples into the
// gradient for every kernel whose size is not 2 or 3.
func TestAuditConvWeightGradTapAlignment(t *testing.T) {
	auditCUDAKernels(t)
	const batch, channels, length, outChannels = 2, 3, 9, 4
	for _, shape := range []struct{ kernel, dilation int }{
		{3, 1}, {3, 2}, {4, 1}, {5, 2}, {1, 2}, {2, 1},
	} {
		kernel, dilation := shape.kernel, shape.dilation
		x := make([]float32, batch*channels*length)
		dy := make([]float32, batch*outChannels*length)
		for i := range x {
			x[i] = float32(math.Sin(float64(i) * 0.37))
		}
		for i := range dy {
			dy[i] = float32(math.Cos(float64(i) * 0.21))
		}
		want := referenceConvWeightGrad(x, dy, batch, channels, length, outChannels, kernel, dilation)
		deviceX := upload(t, x)
		deviceDY := upload(t, dy)
		deviceDW, err := cuda.Alloc(len(want) * 4)
		if err != nil {
			t.Fatal(err)
		}
		defer deviceDW.Free()
		defer deviceX.Free()
		defer deviceDY.Free()
		if err := ConvWeightGrad(deviceX, deviceDY, deviceDW, batch, channels, length, outChannels, kernel, dilation); err != nil {
			t.Fatal(err)
		}
		if err := cuda.Synchronize(); err != nil {
			t.Fatal(err)
		}
		got := download(t, deviceDW, len(want))
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("kernel=%d dilation=%d dw[%d]: got %v want %v", kernel, dilation, i, got[i], want[i])
			}
		}
	}
}

// TestAuditEmptyElementwiseLaunches checks that an empty tensor does not turn
// into an invalid launch. cuLaunchKernel rejects a zero-sized grid, which
// would turn an empty batch into a hard CUDA error.
func TestAuditEmptyElementwiseLaunches(t *testing.T) {
	auditCUDAKernels(t)
	empty, err := cuda.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Free()
	out, err := cuda.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := Tanh(empty, out, 0); err != nil {
		t.Fatalf("tanh with zero elements: %v", err)
	}
	if err := AddInto(out, empty, 0); err != nil {
		t.Fatalf("add into with zero elements: %v", err)
	}
	if err := AddPair(out, empty, empty, 0); err != nil {
		t.Fatalf("add pair with zero elements: %v", err)
	}
	if err := TanhBackward(empty, empty, out, 0); err != nil {
		t.Fatalf("tanh backward with zero elements: %v", err)
	}
	if err := SumSquares(empty, out, 0); err != nil {
		t.Fatalf("sum of squares with zero elements: %v", err)
	}
	if err := AddBiasColumns(out, empty, 0, 4); err != nil {
		t.Fatalf("add bias columns with zero elements: %v", err)
	}
	if err := Transpose12(empty, out, 0, 2, 2); err != nil {
		t.Fatalf("transpose with zero elements: %v", err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
}

// TestAuditActivationBackwardRequiresGradient checks that a backward activation
// without a gradient buffer reports an error instead of reading a null device
// pointer, which would poison the whole CUDA context.
func TestAuditActivationBackwardRequiresGradient(t *testing.T) {
	auditCUDAKernels(t)
	x, err := cuda.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	out, err := cuda.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := Activation(x, nil, out, 4, 0, 0, true); err == nil {
		t.Fatal("activation backward without a gradient buffer was accepted")
	}
}

// TestAuditAdamWWithoutClipping checks that a non-positive maxNorm disables the
// gradient clipping instead of zeroing every gradient.
func TestAuditAdamWWithoutClipping(t *testing.T) {
	auditCUDAKernels(t)
	const count = 4
	values := []float32{1, 2, 3, 4}
	gradients := []float32{0.5, -0.25, 1, 2}
	deviceP := upload(t, values)
	deviceG := upload(t, gradients)
	deviceM, err := cuda.Alloc(count * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer deviceM.Free()
	deviceV, err := cuda.Alloc(count * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer deviceV.Free()
	norm, err := cuda.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer norm.Free()
	if err := norm.Memset(0, 4); err != nil {
		t.Fatal(err)
	}
	if err := AdamWUpdate(deviceP, deviceG, deviceM, deviceV, norm, count, .9, .999, .01, 1, 0, 1e-8, 0); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceP, count)
	for i := range got {
		if got[i] == values[i] {
			t.Fatalf("parameter %d did not change with clipping disabled: %v", i, got)
		}
	}
}
