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

func BenchmarkSgemmTransposeA(b *testing.B) {
	benchSetup(b)
	rows := benchBatch * benchLength
	a := benchBuffer(b, rows*benchChannels)
	bb := benchBuffer(b, rows*benchChannels)
	c := benchBuffer(b, benchChannels*benchChannels)
	blas, err := NewBlas()
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { blas.Destroy() })
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := blas.SgemmRowMajorTransposeA(benchChannels, benchChannels, rows, 1, a.Pointer(), benchChannels, bb.Pointer(), benchChannels, 0, c.Pointer(), benchChannels); err != nil {
			b.Fatal(err)
		}
	}
	if err := Synchronize(); err != nil {
		b.Fatal(err)
	}
}
