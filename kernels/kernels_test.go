package kernels

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

func floatsToBytes(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}

func bytesToFloats(data []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4)
}

func upload(t *testing.T, values []float32) *cuda.Buffer {
	t.Helper()
	buffer, err := cuda.Alloc(len(values) * 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := buffer.CopyFromHost(floatsToBytes(values)); err != nil {
		t.Fatalf("copy from host: %v", err)
	}
	return buffer
}

func download(t *testing.T, buffer *cuda.Buffer, count int) []float32 {
	t.Helper()
	data := make([]byte, count*4)
	if err := buffer.CopyToHost(data); err != nil {
		t.Fatalf("copy to host: %v", err)
	}
	return bytesToFloats(data)
}

func loadErrorText() string {
	if _, err := cuda.DeviceCount(); err != nil {
		return err.Error()
	}
	return ""
}

// TestElementwiseKernels checks the library elementwise kernels against
// closed-form values. The kernels are compiled and launched through the same
// path as the rest of the package.
func TestElementwiseKernels(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	input := []float32{-2, -0.5, 0, 0.5, 2}
	other := []float32{1, 2, 3, 4, 5}
	deviceInput := upload(t, input)
	defer deviceInput.Free()
	deviceOther := upload(t, other)
	defer deviceOther.Free()
	tanhOut, err := cuda.Alloc(len(input) * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer tanhOut.Free()
	if err := Tanh(deviceInput, tanhOut, len(input)); err != nil {
		t.Fatal(err)
	}
	sumOut, err := cuda.Alloc(len(input) * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer sumOut.Free()
	if err := AddPair(sumOut, deviceInput, deviceOther, len(input)); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	gotTanh := download(t, tanhOut, len(input))
	for i, v := range input {
		want := float32(math.Tanh(float64(v)))
		if math.Abs(float64(gotTanh[i]-want)) > 1e-6 {
			t.Errorf("tanh[%d]: got %v want %v", i, gotTanh[i], want)
		}
	}
	gotSum := download(t, sumOut, len(input))
	for i := range input {
		if gotSum[i] != input[i]+other[i] {
			t.Errorf("add[%d]: got %v want %v", i, gotSum[i], input[i]+other[i])
		}
	}
}
