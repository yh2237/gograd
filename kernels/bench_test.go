package kernels

import (
	"runtime"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

const (
	benchBatch       = 64
	benchChannels    = 24
	benchLength      = 200
	benchOutChannels = 24
	benchKernel      = 3
	benchDilation    = 1
)

func benchSetup(b *testing.B) {
	b.Helper()
	if !cuda.Available() {
		b.Skip("cuda unavailable")
	}
	runtime.LockOSThread()
	b.Cleanup(runtime.UnlockOSThread)
	if err := cuda.SetDevice(0); err != nil {
		b.Fatal(err)
	}
}

func benchBuffer(b *testing.B, count int) *cuda.Buffer {
	b.Helper()
	buffer, err := cuda.Alloc(count * 4)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { buffer.Free() })
	return buffer
}

func BenchmarkConv1dForward(b *testing.B) {
	benchSetup(b)
	x := benchBuffer(b, benchBatch*benchChannels*benchLength)
	w := benchBuffer(b, benchOutChannels*benchChannels*benchKernel)
	bias := benchBuffer(b, benchOutChannels)
	out := benchBuffer(b, benchBatch*benchOutChannels*benchLength)
	blas, err := cuda.NewBlas()
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { blas.Destroy() })
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Conv1dForward(blas, x, w, bias, out, benchBatch, benchChannels, benchLength, benchOutChannels, benchKernel, benchDilation); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkConvWeightGrad(b *testing.B) {
	benchSetup(b)
	x := benchBuffer(b, benchBatch*benchChannels*benchLength)
	dy := benchBuffer(b, benchBatch*benchOutChannels*benchLength)
	dw := benchBuffer(b, benchOutChannels*benchChannels*benchKernel)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvWeightGrad(x, dy, dw, benchBatch, benchChannels, benchLength, benchOutChannels, benchKernel, benchDilation); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkConvInputGrad(b *testing.B) {
	benchSetup(b)
	dy := benchBuffer(b, benchBatch*benchOutChannels*benchLength)
	w := benchBuffer(b, benchOutChannels*benchChannels*benchKernel)
	dx := benchBuffer(b, benchBatch*benchChannels*benchLength)
	blas, err := cuda.NewBlas()
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { blas.Destroy() })
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvInputGrad(blas, dy, w, dx, benchBatch, benchChannels, benchLength, benchOutChannels, benchKernel, benchDilation); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkTranspose12(b *testing.B) {
	benchSetup(b)
	d0, d1, d2 := benchBatch, benchLength, benchChannels
	input := benchBuffer(b, d0*d1*d2)
	output := benchBuffer(b, d0*d1*d2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Transpose12(input, output, d0, d1, d2); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkAddInto(b *testing.B) {
	benchSetup(b)
	count := benchBatch * benchLength * benchChannels
	dst := benchBuffer(b, count)
	src := benchBuffer(b, count)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AddInto(dst, src, count); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkAddTanh(b *testing.B) {
	benchSetup(b)
	count := benchBatch * benchLength * benchChannels
	a := benchBuffer(b, count)
	bb := benchBuffer(b, count)
	out := benchBuffer(b, count)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AddTanh(a, bb, out, count); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkConvBiasGrad(b *testing.B) {
	benchSetup(b)
	dy := benchBuffer(b, benchBatch*benchOutChannels*benchLength)
	db := benchBuffer(b, benchOutChannels)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvBiasGrad(dy, db, benchBatch, benchOutChannels, benchLength); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkTanhBackward(b *testing.B) {
	benchSetup(b)
	rows := benchBatch * benchLength
	count := rows * benchChannels
	a := benchBuffer(b, count)
	dout := benchBuffer(b, count)
	din := benchBuffer(b, count)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TanhBackward(a, dout, din, count); err != nil {
			b.Fatal(err)
		}
	}
	if err := cuda.Synchronize(); err != nil {
		b.Fatal(err)
	}
}
