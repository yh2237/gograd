//go:build linux && !android && amd64

#include "textflag.h"

TEXT ·sysvCallPC(SB), NOSPLIT, $0-8
	LEAQ sysvCall<>(SB), AX
	MOVQ AX, ret+0(FP)
	RET

TEXT dlopenTramp<>(SB), NOSPLIT|NOFRAME, $0
	JMP x_dlopen(SB)

TEXT dlsymTramp<>(SB), NOSPLIT|NOFRAME, $0
	JMP x_dlsym(SB)

TEXT dlcloseTramp<>(SB), NOSPLIT|NOFRAME, $0
	JMP x_dlclose(SB)

TEXT dlerrorTramp<>(SB), NOSPLIT|NOFRAME, $0
	JMP x_dlerror(SB)

TEXT ·dlopenAddr(SB), NOSPLIT, $0-8
	LEAQ dlopenTramp<>(SB), AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·dlsymAddr(SB), NOSPLIT, $0-8
	LEAQ dlsymTramp<>(SB), AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·dlcloseAddr(SB), NOSPLIT, $0-8
	LEAQ dlcloseTramp<>(SB), AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·dlerrorAddr(SB), NOSPLIT, $0-8
	LEAQ dlerrorTramp<>(SB), AX
	MOVQ AX, ret+0(FP)
	RET

// copyCString copies up to dstCap bytes from a NUL-terminated C string at src into dst.
TEXT ·copyCString(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ dstCap+8(FP), CX
	MOVQ src+16(FP), SI
	XORQ AX, AX
	TESTQ SI, SI
	JZ done
loop:
	CMPQ AX, CX
	JGE done
	MOVBLZX (SI)(AX*1), DX
	TESTL DX, DX
	JZ done
	MOVB DL, (DI)(AX*1)
	INCQ AX
	JMP loop
done:
	MOVQ AX, ret+24(FP)
	RET

// sysvCall is entered via runtime.cgocall on the system stack with DI = *sysvFrame.
TEXT sysvCall<>(SB), NOSPLIT|NOFRAME, $0
	PUSHQ BP
	MOVQ SP, BP
	PUSHQ BX
	SUBQ $56, SP // 6 stack args (48B) + 8B alignment padding -> SP%16 == 0 before CALL

	MOVQ DI, BX
	MOVQ 0(BX), R11

	MOVQ 56(BX), AX
	MOVQ AX, 0(SP)
	MOVQ 64(BX), AX
	MOVQ AX, 8(SP)
	MOVQ 72(BX), AX
	MOVQ AX, 16(SP)
	MOVQ 80(BX), AX
	MOVQ AX, 24(SP)
	MOVQ 88(BX), AX
	MOVQ AX, 32(SP)
	MOVQ 96(BX), AX
	MOVQ AX, 40(SP)

	MOVQ 8(BX), DI
	MOVQ 16(BX), SI
	MOVQ 24(BX), DX
	MOVQ 32(BX), CX
	MOVQ 40(BX), R8
	MOVQ 48(BX), R9
	XORL AX, AX
	CALL R11

	MOVQ AX, 104(BX)
	ADDQ $56, SP
	POPQ BX
	POPQ BP
	RET
