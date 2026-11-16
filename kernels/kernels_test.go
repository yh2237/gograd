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

func kernelTestSetup(t *testing.T) {
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

func TestAddBiasColumns(t *testing.T) {
	kernelTestSetup(t)
	y := upload(t, []float32{1, 2, 3, 4, 5, 6})
	defer y.Free()
	bias := upload(t, []float32{10, 20, 30})
	defer bias.Free()
	if err := AddBiasColumns(y, bias, 2, 3); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, y, 6)
	want := []float32{11, 22, 33, 14, 25, 36}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("y[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestColumnSum(t *testing.T) {
	kernelTestSetup(t)
	x := upload(t, []float32{1, 2, 3, 4, 5, 6})
	defer x.Free()
	out := upload(t, []float32{0, 0, 0})
	defer out.Free()
	if err := ColumnSum(x, out, 2, 3); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, out, 3)
	want := []float32{5, 7, 9}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestSqrtScalar(t *testing.T) {
	kernelTestSetup(t)
	x := upload(t, []float32{9})
	defer x.Free()
	if err := SqrtScalar(x); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if got := download(t, x, 1)[0]; got != 3 {
		t.Fatalf("sqrt = %v, want 3", got)
	}
}

func TestAdamWUpdate(t *testing.T) {
	kernelTestSetup(t)
	const count = 4
	p := upload(t, []float32{1, -2, .5, 3})
	defer p.Free()
	g := upload(t, []float32{.1, -.2, .3, -.4})
	defer g.Free()
	m := upload(t, []float32{0, 0, 0, 0})
	defer m.Free()
	v := upload(t, []float32{0, 0, 0, 0})
	defer v.Free()
	norm := upload(t, []float32{1})
	defer norm.Free()
	const beta1, beta2, stepSize, biasCorrection2, weightDecay, eps, maxNorm = .9, .999, .001, .999, .01, 1e-8, 1
	if err := AdamWUpdate(p, g, m, v, norm, count, beta1, beta2, stepSize, biasCorrection2, weightDecay, eps, maxNorm); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := download(t, p, count)
	grads := []float32{.1, -.2, .3, -.4}
	params := []float32{1, -2, .5, 3}
	coefficient := float32(math.Min(1, maxNorm/(1+1e-6)))
	for i := range got {
		grad := grads[i] * coefficient
		mv := beta1*0 + (1-beta1)*grad
		vv := beta2*0 + (1-beta2)*grad*grad
		denom := float32(math.Sqrt(float64(vv/biasCorrection2))) + eps
		want := params[i]*(1-weightDecay) - stepSize*mv/denom
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("p[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestEmptyLaunches(t *testing.T) {
	kernelTestSetup(t)
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
	if err := AddBiasColumns(out, empty, 0, 4); err != nil {
		t.Fatalf("add bias columns with zero elements: %v", err)
	}
	if err := ColumnSum(empty, out, 0, 4); err != nil {
		t.Fatalf("column sum with zero elements: %v", err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
}
