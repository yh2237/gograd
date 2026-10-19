package kernels

import (
	"fmt"
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

func TestTranspose12(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
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
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, deviceOut, len(input))
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("transpose[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

// referenceConv1dTimeMajor returns y[o, b*length+t] for time-major
// x [batch,length,channels] without bias. The tap offset is the centered one
// the im2col kernel uses.
func referenceConv1dTimeMajor(x, weight []float32, batch, channels, length, outChannels, kernel, dilation int) []float32 {
	out := make([]float32, outChannels*batch*length)
	for o := 0; o < outChannels; o++ {
		for b := 0; b < batch; b++ {
			for t := 0; t < length; t++ {
				sum := float32(0)
				for c := 0; c < channels; c++ {
					for k := 0; k < kernel; k++ {
						source := t - (kernel/2)*dilation + k*dilation
						if source < 0 || source >= length {
							continue
						}
						sum += weight[(o*channels+c)*kernel+k] * x[(b*length+source)*channels+c]
					}
				}
				out[o*batch*length+b*length+t] = sum
			}
		}
	}
	return out
}

func TestConv1dForward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	const batch, channels, length, outChannels = 2, 3, 8, 4
	for _, shape := range []struct{ kernel, dilation int }{
		{3, 2}, {3, 1}, {4, 1}, {5, 2}, {1, 1}, {2, 1},
	} {
		kernel, dilation := shape.kernel, shape.dilation
		t.Run(fmt.Sprintf("kernel%d_dilation%d", kernel, dilation), func(t *testing.T) {
			x := make([]float32, batch*length*channels)
			for i := range x {
				x[i] = float32(math.Sin(float64(i) * 0.7))
			}
			weight := make([]float32, outChannels*channels*kernel)
			for i := range weight {
				weight[i] = float32(math.Cos(float64(i) * 0.3))
			}
			want := referenceConv1dTimeMajor(x, weight, batch, channels, length, outChannels, kernel, dilation)

			deviceX := upload(t, x)
			deviceWeight := upload(t, weight)
			columns, err := cuda.Alloc(channels * kernel * batch * length * 4)
			if err != nil {
				t.Fatal(err)
			}
			conv, err := cuda.Alloc(outChannels * batch * length * 4)
			if err != nil {
				t.Fatal(err)
			}
			defer deviceX.Free()
			defer deviceWeight.Free()
			defer columns.Free()
			defer conv.Free()

			blas, err := cuda.NewBlas()
			if err != nil {
				t.Fatal(err)
			}
			defer blas.Destroy()
			if err := Conv1dForwardTo(blas, deviceX, deviceWeight, conv, columns, batch, channels, length, outChannels, kernel, dilation); err != nil {
				t.Fatal(err)
			}
			if err := cuda.Synchronize(); err != nil {
				t.Fatal(err)
			}
			got := download(t, conv, len(want))
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("conv[%d]: got %v want %v", i, got[i], want[i])
				}
			}
		})
	}
}
