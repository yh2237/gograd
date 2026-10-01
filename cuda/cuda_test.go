package cuda

import (
	"math"
	"runtime"
	"testing"
	"unsafe"
)

func floatsToBytes(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}

func bytesToFloats(data []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4)
}

func upload(t *testing.T, values []float32) *Buffer {
	t.Helper()
	buffer, err := Alloc(len(values) * 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := buffer.CopyFromHost(floatsToBytes(values)); err != nil {
		t.Fatalf("copy from host: %v", err)
	}
	return buffer
}

func download(t *testing.T, buffer *Buffer, count int) []float32 {
	t.Helper()
	data := make([]byte, count*4)
	if err := buffer.CopyToHost(data); err != nil {
		t.Fatalf("copy to host: %v", err)
	}
	return bytesToFloats(data)
}

func TestBufferRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	source := []float32{1.5, -2.25, 3.125, 0}
	buffer := upload(t, source)
	defer buffer.Free()
	got := download(t, buffer, len(source))
	for i := range source {
		if got[i] != source[i] {
			t.Fatalf("round trip mismatch at %d: got %v want %v", i, got[i], source[i])
		}
	}
}

func TestSgemmRowMajor(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	a := []float32{1, 2, 3, 4, 5, 6}    // 2x3
	b := []float32{7, 8, 9, 10, 11, 12} // 3x2
	want := []float32{
		a[0]*b[0] + a[1]*b[2] + a[2]*b[4], a[0]*b[1] + a[1]*b[3] + a[2]*b[5],
		a[3]*b[0] + a[4]*b[2] + a[5]*b[4], a[3]*b[1] + a[4]*b[3] + a[5]*b[5],
	}
	deviceA := upload(t, a)
	deviceB := upload(t, b)
	deviceC := upload(t, make([]float32, 4))
	defer deviceA.Free()
	defer deviceB.Free()
	defer deviceC.Free()

	blas, err := NewBlas()
	if err != nil {
		t.Fatal(err)
	}
	defer blas.Destroy()
	if err := blas.SgemmRowMajor(2, 2, 3, 1, deviceA.Pointer(), 3, deviceB.Pointer(), 2, 0, deviceC.Pointer(), 2); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceC, 4)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5*math.Max(1, math.Abs(float64(want[i]))) {
			t.Errorf("sgemm[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

const testKernelSource = `
extern "C" __global__ void vec_add(const float* a, const float* b, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void apply_tanh(const float* x, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = tanhf(x[i]);
}
`

func launchUnary(t *testing.T, kernel *Kernel, input *Buffer, n int) *Buffer {
	t.Helper()
	output, err := Alloc(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	inputAddress := input.Pointer()
	outputAddress := output.Pointer()
	size := int32(n)
	args := []unsafe.Pointer{unsafe.Pointer(&inputAddress), unsafe.Pointer(&outputAddress), unsafe.Pointer(&size)}
	grid := [3]int{(n + 255) / 256, 1, 1}
	if err := kernel.Launch(grid, [3]int{256, 1, 1}, 0, nil, args); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	return output
}

func TestKernelVectorAdd(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	kernel, err := LoadKernel(testKernelSource, "vec_add")
	if err != nil {
		t.Fatal(err)
	}
	defer kernel.Close()

	a := []float32{1, 2, 3, 4, 5}
	b := []float32{10, 20, 30, 40, 50}
	deviceA := upload(t, a)
	deviceB := upload(t, b)
	output, err := Alloc(len(a) * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer deviceA.Free()
	defer deviceB.Free()
	defer output.Free()

	aAddr, bAddr, outAddr := deviceA.Pointer(), deviceB.Pointer(), output.Pointer()
	size := int32(len(a))
	args := []unsafe.Pointer{unsafe.Pointer(&aAddr), unsafe.Pointer(&bAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&size)}
	if err := kernel.Launch([3]int{1, 1, 1}, [3]int{256, 1, 1}, 0, nil, args); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, output, len(a))
	for i := range a {
		if got[i] != a[i]+b[i] {
			t.Errorf("add[%d]: got %v want %v", i, got[i], a[i]+b[i])
		}
	}
}

func TestKernelTanh(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	kernel, err := LoadKernel(testKernelSource, "apply_tanh")
	if err != nil {
		t.Fatal(err)
	}
	defer kernel.Close()

	input := []float32{-2, -0.5, 0, 0.5, 2}
	deviceInput := upload(t, input)
	defer deviceInput.Free()
	output := launchUnary(t, kernel, deviceInput, len(input))
	defer output.Free()
	got := download(t, output, len(input))
	for i := range input {
		want := float32(math.Tanh(float64(input[i])))
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Errorf("tanh[%d]: got %v want %v", i, got[i], want)
		}
	}
}

func loadErrorText() string {
	if _, err := DeviceCount(); err != nil {
		return err.Error()
	}
	return ""
}
