#include "textflag.h"

TEXT ·hasAVX2FMA(SB), NOSPLIT, $0-1
	MOVL $1, AX
	CPUID
	MOVL CX, R8
	ANDL $(1<<12 | 1<<27 | 1<<28), R8
	CMPL R8, $(1<<12 | 1<<27 | 1<<28)
	JNE noavx
	XORL CX, CX
	XGETBV
	ANDL $6, AX
	CMPL AX, $6
	JNE noavx
	MOVL $7, AX
	XORL CX, CX
	CPUID
	TESTL $(1<<5), BX
	JZ noavx
	MOVB $1, ret+0(FP)
	RET
noavx:
	MOVB $0, ret+0(FP)
	RET

// c points at the first output of a 4x4 tile. a and b contain packed
// depth-major groups of four floats. stride is measured in floats.
TEXT ·kernel4x4AVX(SB), NOSPLIT, $0-41
	MOVQ c+0(FP), CX
	MOVQ a+8(FP), AX
	MOVQ b+16(FP), BX
	MOVQ stride+24(FP), DX
	SHLQ $2, DX
	MOVQ k+32(FP), R8
	MOVBLZX add+40(FP), R9
	VXORPS X0, X0, X0
	VXORPS X1, X1, X1
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	TESTQ R9, R9
	JZ loopcheck
	VMOVUPS (CX), X0
	VMOVUPS (CX)(DX*1), X1
	LEAQ (CX)(DX*2), R10
	VMOVUPS (R10), X2
	ADDQ DX, R10
	VMOVUPS (R10), X3
loopcheck:
	TESTQ R8, R8
	JZ finish
loop:
	VMOVUPS (BX), X4
	VBROADCASTSS (AX), X5
	VFMADD231PS X4, X5, X0
	VBROADCASTSS 4(AX), X5
	VFMADD231PS X4, X5, X1
	VBROADCASTSS 8(AX), X5
	VFMADD231PS X4, X5, X2
	VBROADCASTSS 12(AX), X5
	VFMADD231PS X4, X5, X3
	ADDQ $16, AX
	ADDQ $16, BX
	DECQ R8
	JNZ loop
finish:
	VMOVUPS X0, (CX)
	VMOVUPS X1, (CX)(DX*1)
	LEAQ (CX)(DX*2), R10
	VMOVUPS X2, (R10)
	ADDQ DX, R10
	VMOVUPS X3, (R10)
	RET
