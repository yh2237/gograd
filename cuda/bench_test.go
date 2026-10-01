package cuda

import (
	"runtime"
	"testing"
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
	if !Available() {
		b.Skip("cuda unavailable")
	}
	runtime.LockOSThread()
	b.Cleanup(runtime.UnlockOSThread)
	if err := SetDevice(0); err != nil {
		b.Fatal(err)
	}
}

func benchBuffer(b *testing.B, count int) *Buffer {
	b.Helper()
	buffer, err := Alloc(count * 4)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { buffer.Free() })
	return buffer
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
	if err := Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkConvInputGrad(b *testing.B) {
	benchSetup(b)
	dy := benchBuffer(b, benchBatch*benchOutChannels*benchLength)
	w := benchBuffer(b, benchOutChannels*benchChannels*benchKernel)
	dx := benchBuffer(b, benchBatch*benchChannels*benchLength)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvInputGrad(dy, w, dx, benchBatch, benchChannels, benchLength, benchOutChannels, benchKernel, benchDilation); err != nil {
			b.Fatal(err)
		}
	}
	if err := Synchronize(); err != nil {
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
	if err := Synchronize(); err != nil {
		b.Fatal(err)
	}
}
