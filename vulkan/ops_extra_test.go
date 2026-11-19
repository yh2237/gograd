package vulkan

import (
	"errors"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func TestVulkanExtraOps(t *testing.T) {
	if !Available() {
		t.Skipf("Vulkan unavailable: %v", Err())
	}
	b := NewBackend()
	defer func() {
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	aVals := []float32{-3.5, -1, 0, 0.25, 2.5, 11.75}
	bVals := []float32{1.25, -4, 0, 2, -0.5, 3.25}
	aBuf, err := b.Upload(aVals)
	if err != nil {
		t.Fatal(err)
	}
	defer aBuf.Close()
	bBuf, err := b.Upload(bVals)
	if err != nil {
		t.Fatal(err)
	}
	defer bBuf.Close()
	out, err := b.Alloc(len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	if err := b.BinaryOp(gpu.BinarySub, out, aBuf, bBuf, len(aVals)); err != nil {
		t.Fatal(err)
	}
	got, err := b.Download(out, len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	for i := range aVals {
		want := aVals[i] - bVals[i]
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("BinarySub[%d] = %v, want %v", i, got[i], want)
		}
	}

	if err := b.ScalarOp(gpu.ScalarAdd, out, aBuf, len(aVals), 2.5); err != nil {
		t.Fatal(err)
	}
	got, err = b.Download(out, len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	for i := range aVals {
		want := aVals[i] + 2.5
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("ScalarAdd[%d] = %v, want %v", i, got[i], want)
		}
	}

	if err := b.ScalarOp(gpu.ScalarSub, out, aBuf, len(aVals), 1.5); err != nil {
		t.Fatal(err)
	}
	got, err = b.Download(out, len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	for i := range aVals {
		want := aVals[i] - 1.5
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("ScalarSub[%d] = %v, want %v", i, got[i], want)
		}
	}

	if err := b.ScalarOp(gpu.ScalarDiv, out, aBuf, len(aVals), 4); err != nil {
		t.Fatal(err)
	}
	got, err = b.Download(out, len(aVals))
	if err != nil {
		t.Fatal(err)
	}
	for i := range aVals {
		want := aVals[i] / 4
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("ScalarDiv[%d] = %v, want %v", i, got[i], want)
		}
	}

	for _, op := range []gpu.BinaryOp{gpu.BinaryDiv, gpu.BinaryMax, gpu.BinaryMin} {
		if err := b.BinaryOp(op, out, aBuf, bBuf, len(aVals)); !errors.Is(err, gpu.ErrUnsupported) {
			t.Fatalf("BinaryOp(%d) = %v, want ErrUnsupported", op, err)
		}
	}
	if err := b.ScalarOp(gpu.ScalarPow, out, aBuf, len(aVals), 2); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatalf("ScalarPow = %v, want ErrUnsupported", err)
	}
	if err := b.ScalarOp(gpu.ScalarDiv, out, aBuf, len(aVals), 0); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatalf("ScalarDiv(0) = %v, want ErrUnsupported", err)
	}
	if err := b.ScalarOp(gpu.ScalarDiv, out, aBuf, len(aVals), 0.03); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatalf("ScalarDiv(0.03) = %v, want ErrUnsupported", err)
	}
	if err := b.BinaryOp(gpu.BinarySub, aBuf, aBuf, bBuf, len(aVals)); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatalf("BinarySub aliasing = %v, want ErrInvalidArgument", err)
	}
	if err := b.ScalarOp(gpu.ScalarAdd, aBuf, aBuf, len(aVals), 1); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatalf("ScalarAdd aliasing = %v, want ErrInvalidArgument", err)
	}

	empty, err := b.Alloc(0)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	emptyOut, err := b.Alloc(0)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyOut.Close()
	if err := b.BinaryOp(gpu.BinarySub, emptyOut, empty, empty, 0); err != nil {
		t.Fatal(err)
	}
	if err := b.ScalarOp(gpu.ScalarAdd, emptyOut, empty, 0, 1); err != nil {
		t.Fatal(err)
	}
	if err := b.ScalarOp(gpu.ScalarDiv, emptyOut, empty, 0, 2); err != nil {
		t.Fatal(err)
	}
}
