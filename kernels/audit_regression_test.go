package kernels

import (
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
