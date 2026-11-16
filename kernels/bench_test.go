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
