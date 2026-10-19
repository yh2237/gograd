//go:build linux && !android && amd64

#include "textflag.h"

TEXT ·Address(SB), NOSPLIT, $0-8
	LEAQ probe<>(SB), AX
	MOVQ AX, ret+0(FP)
	RET

// This is entered using the native SysV ABI, not Go's ABI. All registers used
// are caller-saved, and SP is not adjusted (four arguments reside on the stack).
TEXT probe<>(SB), NOSPLIT|NOFRAME, $0
	MOVQ $0x1020304050607080, AX
	ROLQ $7, AX
	XORQ SI, AX
	ROLQ $7, AX
	XORQ DX, AX
	ROLQ $7, AX
	XORQ CX, AX
	ROLQ $7, AX
	XORQ R8, AX
	ROLQ $7, AX
	XORQ R9, AX
	ROLQ $7, AX
	XORQ 8(SP), AX
	ROLQ $7, AX
	XORQ 16(SP), AX
	ROLQ $7, AX
	XORQ 24(SP), AX
	ROLQ $7, AX
	XORQ 32(SP), AX
	MOVQ AX, 0(DI)
	MOVQ SP, R10
	ANDQ $15, R10
	MOVQ R10, 8(DI)
	RET
