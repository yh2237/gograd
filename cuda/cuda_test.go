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

func loadErrorText() string {
	if _, err := DeviceCount(); err != nil {
		return err.Error()
	}
	return ""
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

const graphKernelSource = `
extern "C" __global__ void add_arrays(const float* a, const float* b, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = a[i] + b[i];
}
`

func launchAdd(t *testing.T, kernel *Kernel, a, b, out *Buffer, n int) {
	t.Helper()
	aAddr, bAddr, outAddr := a.Pointer(), b.Pointer(), out.Pointer()
	size := int32(n)
	args := []unsafe.Pointer{unsafe.Pointer(&aAddr), unsafe.Pointer(&bAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&size)}
	if err := kernel.Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, nil, args); err != nil {
		t.Fatal(err)
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

	kernel, err := LoadKernel(graphKernelSource, "add_arrays")
	if err != nil {
		t.Fatal(err)
	}
	defer kernel.Close()

	stream, err := NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	launchAdd(t, kernel, a, b, out, n)
	if err := Synchronize(); err != nil {
		t.Fatal(err)
	}

	graph, err := Capture(stream, func() error {
		launchAdd(t, kernel, a, b, out, n)
		return nil
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
