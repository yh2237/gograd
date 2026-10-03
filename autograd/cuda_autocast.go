package autograd

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// BF16Autocast converts GEMM inputs on the device, computes with tensor cores
// and FP32 accumulation, and leaves outputs, gradients and master weights FP32.
// Set this only between training steps; the switch is process-wide.
var BF16Autocast bool

func castBF16(src uintptr, count int) *cuda.Buffer {
	b := allocBF16(count)
	dst, n := ptr(b), int32(count)
	launch("f32_to_bf16", count, unsafe.Pointer(&src), unsafe.Pointer(&dst), unsafe.Pointer(&n))
	return b
}

func allocBF16(count int) *cuda.Buffer {
	b, err := cuda.Alloc(count * 2)
	if err != nil {
		panic(err)
	}
	return b
}

func (t *Tensor) invalidateBF16() {
	if t.bf16Buf != nil {
		if t.bf16Owner {
			t.bf16Buf.Free()
		}
		t.bf16Buf = nil
		t.bf16Owner = false
	}
}
func (t *Tensor) ensureBF16() *cuda.Buffer {
	if t.bf16Buf == nil {
		t.bf16Buf = castBF16(ptr(t.buf), t.Numel())
		t.bf16Owner = true
	}
	return t.bf16Buf
}

// gemmBatchedPrepared receives BF16 operand buffers created by a producer or
// cached on a tensor. This avoids a cast per GEMM while preserving FP32 output.
func gemmBatchedPrepared(m, n, k int, ta, tb bool, a, a16 uintptr, as int64, b, b16 uintptr, bs int64, c uintptr, cs int64, count int, beta float32) error {
	if !BF16Autocast {
		return blas().SgemmRowMajorStridedBatched(m, n, k, ta, tb, 1, a, as, b, bs, beta, c, cs, count)
	}
	return blas().GemmRowMajorStridedBF16(m, n, k, ta, tb, 1, a16, as, b16, bs, beta, c, cs, count)
}
func gemmSinglePrepared(m, n, k int, ta, tb bool, a, a16, b, b16, c uintptr, beta float32) error {
	if !BF16Autocast {
		return gemmSingle(m, n, k, ta, tb, a, b, c, beta)
	}
	return blas().GemmRowMajorBF16(m, n, k, ta, tb, 1, a16, b16, beta, c)
}

func gemmSingle(m, n, k int, ta, tb bool, a, b, c uintptr, beta float32) error {
	if !BF16Autocast {
		switch {
		case ta && !tb:
			return blas().SgemmRowMajorTransposeA(m, n, k, 1, a, func() int {
				if ta {
					return m
				}
				return k
			}(), b, n, beta, c, n)
		case !ta && tb:
			return blas().SgemmRowMajorNT(m, n, k, 1, a, k, b, k, beta, c, n)
		case !ta && !tb:
			return blas().SgemmRowMajor(m, n, k, 1, a, k, b, n, beta, c, n)
		default:
			return blas().SgemmRowMajorStridedBatched(m, n, k, ta, tb, 1, a, 0, b, 0, beta, c, 0, 1)
		}
	}
	ab, bb := castBF16(a, m*k), castBF16(b, k*n)
	defer ab.Free()
	defer bb.Free()
	return blas().GemmRowMajorBF16(m, n, k, ta, tb, 1, ptr(ab), ptr(bb), beta, c)
}

func gemmBatched(m, n, k int, ta, tb bool, a uintptr, as int64, b uintptr, bs int64, c uintptr, cs int64, count int, beta float32) error {
	if !BF16Autocast {
		return blas().SgemmRowMajorStridedBatched(m, n, k, ta, tb, 1, a, as, b, bs, beta, c, cs, count)
	}
	ac, bc := m*k, k*n
	if as != 0 {
		ac = int(as) * count
	}
	if bs != 0 {
		bc = int(bs) * count
	}
	ab, bb := castBF16(a, ac), castBF16(b, bc)
	defer ab.Free()
	defer bb.Free()
	return blas().GemmRowMajorStridedBF16(m, n, k, ta, tb, 1, ptr(ab), as, ptr(bb), bs, beta, c, cs, count)
}
