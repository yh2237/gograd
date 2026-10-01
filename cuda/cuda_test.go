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

func TestSgemmRowMajorNT(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	a := []float32{1, 2, 3, 4, 5, 6}    // 2x3
	b := []float32{7, 8, 9, 10, 11, 12} // 2x3
	want := []float32{
		a[0]*b[0] + a[1]*b[1] + a[2]*b[2], a[0]*b[3] + a[1]*b[4] + a[2]*b[5],
		a[3]*b[0] + a[4]*b[1] + a[5]*b[2], a[3]*b[3] + a[4]*b[4] + a[5]*b[5],
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
	if err := blas.SgemmRowMajorNT(2, 2, 3, 1, deviceA.Pointer(), 3, deviceB.Pointer(), 3, 0, deviceC.Pointer(), 2); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceC, 4)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("gemm-nt[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestSgemmRowMajorTransposeA(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	a := []float32{1, 2, 3, 4, 5, 6} // 2x3, treated as a^T [3,2]
	bb := []float32{7, 8, 9, 10}     // 2x2
	want := []float32{
		a[0]*bb[0] + a[3]*bb[2], a[0]*bb[1] + a[3]*bb[3],
		a[1]*bb[0] + a[4]*bb[2], a[1]*bb[1] + a[4]*bb[3],
		a[2]*bb[0] + a[5]*bb[2], a[2]*bb[1] + a[5]*bb[3],
	}
	deviceA := upload(t, a)
	deviceB := upload(t, bb)
	deviceC := upload(t, make([]float32, 6))
	defer deviceA.Free()
	defer deviceB.Free()
	defer deviceC.Free()
	blas, err := NewBlas()
	if err != nil {
		t.Fatal(err)
	}
	defer blas.Destroy()
	if err := blas.SgemmRowMajorTransposeA(3, 2, 2, 1, deviceA.Pointer(), 3, deviceB.Pointer(), 2, 0, deviceC.Pointer(), 2); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceC, 6)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("gemm-ta[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestTranspose12(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	// [2,3,2] -> [2,2,3]
	d0, d1, d2 := 2, 3, 2
	input := make([]float32, d0*d1*d2)
	for i := range input {
		input[i] = float32(i)
	}
	want := make([]float32, len(input))
	for b := 0; b < d0; b++ {
		for j := 0; j < d2; j++ {
			for k := 0; k < d1; k++ {
				want[(b*d2+j)*d1+k] = input[(b*d1+k)*d2+j]
			}
		}
	}
	deviceIn := upload(t, input)
	deviceOut := upload(t, make([]float32, len(input)))
	defer deviceIn.Free()
	defer deviceOut.Free()
	if err := Transpose12(deviceIn, deviceOut, d0, d1, d2); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceOut, len(input))
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("transpose[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

func referenceConv1d(x, weight, bias []float32, batch, channels, length, outChannels, kernel, dilation int) []float32 {
	out := make([]float32, batch*outChannels*length)
	for b := 0; b < batch; b++ {
		for o := 0; o < outChannels; o++ {
			for t := 0; t < length; t++ {
				sum := bias[o]
				for c := 0; c < channels; c++ {
					for k := 0; k < kernel; k++ {
						source := t - dilation + k*dilation
						if source < 0 || source >= length {
							continue
						}
						sum += weight[(o*channels+c)*kernel+k] * x[(b*channels+c)*length+source]
					}
				}
				out[(b*outChannels+o)*length+t] = sum
			}
		}
	}
	return out
}

func TestConv1dForward(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	const batch, channels, length, outChannels, kernel, dilation = 2, 3, 8, 4, 3, 2
	x := make([]float32, batch*channels*length)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.7))
	}
	weight := make([]float32, outChannels*channels*kernel)
	for i := range weight {
		weight[i] = float32(math.Cos(float64(i) * 0.3))
	}
	bias := []float32{0.1, -0.2, 0.3, 0.05}
	want := referenceConv1d(x, weight, bias, batch, channels, length, outChannels, kernel, dilation)

	deviceX := upload(t, x)
	deviceWeight := upload(t, weight)
	deviceBias := upload(t, bias)
	deviceOut, err := Alloc(batch * outChannels * length * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer deviceX.Free()
	defer deviceWeight.Free()
	defer deviceBias.Free()
	defer deviceOut.Free()

	blas, err := NewBlas()
	if err != nil {
		t.Fatal(err)
	}
	defer blas.Destroy()
	if err := Conv1dForward(blas, deviceX, deviceWeight, deviceBias, deviceOut, batch, channels, length, outChannels, kernel, dilation); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceOut, len(want))
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("conv[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestGraphCapture(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	const n = 1024
	aHost := make([]float32, n)
	bHost := make([]float32, n)
	for i := 0; i < n; i++ {
		aHost[i] = 1
		bHost[i] = 2
	}
	a := upload(t, aHost)
	b := upload(t, bHost)
	out := upload(t, make([]float32, n))
	defer a.Free()
	defer b.Free()
	defer out.Free()

	stream, err := NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	if err := AddPair(out, a, b, n); err != nil {
		t.Fatal(err)
	}
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}

	graph, err := Capture(stream, func() error {
		return AddPair(out, a, b, n)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()

	if err := graph.Launch(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, out, n)
	for i := range got {
		if got[i] != 3 {
			t.Fatalf("replay[%d]: got %v want 3", i, got[i])
		}
	}

	// The graph reads the buffers at replay time, so changing an input shows up
	// on the next launch.
	for i := 0; i < n; i++ {
		aHost[i] = 5
	}
	if err := a.CopyFromHost(floatsToBytes(aHost)); err != nil {
		t.Fatal(err)
	}
	if err := graph.Launch(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got = download(t, out, n)
	for i := range got {
		if got[i] != 7 {
			t.Fatalf("replay after update[%d]: got %v want 7", i, got[i])
		}
	}
}

func loadErrorText() string {
	if _, err := DeviceCount(); err != nil {
		return err.Error()
	}
	return ""
}
