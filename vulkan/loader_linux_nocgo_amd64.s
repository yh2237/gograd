//go:build linux && !android && amd64 && !cgo

#include "textflag.h"

DATA savedSetg<>(SB)/8, $0
GLOBL savedSetg<>(SB), NOPTR, $8

// cgoInitSym is invoked by runtime.rt0_go before Go runtime initialization:
// DI = *runtime.g (g0), SI = setg_gcc.
TEXT ·cgoInitSym(SB), NOSPLIT|NOFRAME, $0
	PUSHQ BP
	MOVQ SP, BP
	PUSHQ R12
	// Frame layout (104 bytes, SP%16 == 0 before CALL):
	//   0(SP)..63(SP):  pthread_attr_t (56 bytes on glibc x86_64)
	//   64(SP):         stack addr (void*)
	//   72(SP):         stack size (size_t)
	SUBQ $104, SP

	MOVQ DI, R12
	MOVQ SI, savedSetg<>(SB)

	LEAQ 0(SP), DI
	CALL x_pthread_attr_init(SB)
	TESTL AX, AX
	JNZ init_done

	CALL x_pthread_self(SB)
	MOVQ AX, DI
	LEAQ 0(SP), SI
	CALL x_pthread_getattr_np(SB)
	TESTL AX, AX
	JNZ init_destroy

	LEAQ 0(SP), DI
	LEAQ 64(SP), SI
	LEAQ 72(SP), DX
	CALL x_pthread_attr_getstack(SB)
	TESTL AX, AX
	JNZ init_destroy

	MOVQ 64(SP), AX
	TESTQ AX, AX
	JZ init_destroy
	MOVQ AX, 0(R12) // g0.stack.lo

init_destroy:
	LEAQ 0(SP), DI
	CALL x_pthread_attr_destroy(SB)

init_done:
	ADDQ $104, SP
	POPQ R12
	POPQ BP
	RET

// cgoThreadStartSym is invoked by runtime.newosproc with DI = *threadStart ({g, tls, fn}).
TEXT ·cgoThreadStartSym(SB), NOSPLIT|NOFRAME, $0
	PUSHQ BP
	MOVQ SP, BP
	PUSHQ R12
	PUSHQ R13
	PUSHQ R14
	// Frame layout (344 bytes, SP%16 == 0 before CALL):
	//   0(SP)..63(SP):    pthread_attr_t
	//   64(SP)..191(SP):  sigset_t ign (128 bytes)
	//   192(SP)..319(SP): sigset_t oset (128 bytes)
	//   320(SP):          pthread_t tid
	SUBQ $344, SP

	MOVQ DI, R12 // original *threadStart

	MOVQ $24, DI
	CALL x_malloc(SB)
	TESTQ AX, AX
	JZ start_abort
	MOVQ AX, R13 // heap copy *threadStart

	MOVQ 0(R12), CX
	MOVQ CX, 0(R13)
	MOVQ 8(R12), CX
	MOVQ CX, 8(R13)
	MOVQ 16(R12), CX
	MOVQ CX, 16(R13)

	LEAQ 0(SP), DI
	CALL x_pthread_attr_init(SB)
	TESTL AX, AX
	JNZ start_free_abort

	LEAQ 0(SP), DI
	MOVQ $1, SI // PTHREAD_CREATE_DETACHED
	CALL x_pthread_attr_setdetachstate(SB)

	LEAQ 64(SP), DI
	CALL x_sigfillset(SB)

	MOVQ $2, DI // SIG_SETMASK
	LEAQ 64(SP), SI
	LEAQ 192(SP), DX
	CALL x_pthread_sigmask(SB)

	LEAQ 320(SP), DI
	LEAQ 0(SP), SI
	LEAQ threadEntry<>(SB), DX
	MOVQ R13, CX
	CALL x_pthread_create(SB)
	MOVL AX, R14

	MOVQ $2, DI // SIG_SETMASK
	LEAQ 192(SP), SI
	XORQ DX, DX
	CALL x_pthread_sigmask(SB)

	LEAQ 0(SP), DI
	CALL x_pthread_attr_destroy(SB)

	TESTL R14, R14
	JNZ start_free_abort

	ADDQ $344, SP
	POPQ R14
	POPQ R13
	POPQ R12
	POPQ BP
	RET

start_free_abort:
	MOVQ R13, DI
	CALL x_free(SB)
start_abort:
	CALL x_abort(SB)
	HLT

TEXT threadEntry<>(SB), NOSPLIT|NOFRAME, $0
	PUSHQ BP
	MOVQ SP, BP
	PUSHQ R12
	PUSHQ R13
	// Frame layout (96 bytes, SP%16 == 0 before CALL):
	//   0(SP)..63(SP): pthread_attr_t
	//   64(SP):        stack addr
	//   72(SP):        stack size
	SUBQ $96, SP

	MOVQ 0(DI), R12  // ts.g
	MOVQ 16(DI), R13 // ts.fn
	CALL x_free(SB)

	// Populate g.stack.lo before entering runtime.mstart.
	LEAQ 0(SP), DI
	CALL x_pthread_attr_init(SB)
	TESTL AX, AX
	JNZ entry_setg
	CALL x_pthread_self(SB)
	MOVQ AX, DI
	LEAQ 0(SP), SI
	CALL x_pthread_getattr_np(SB)
	TESTL AX, AX
	JNZ entry_destroy
	LEAQ 0(SP), DI
	LEAQ 64(SP), SI
	LEAQ 72(SP), DX
	CALL x_pthread_attr_getstack(SB)
	TESTL AX, AX
	JNZ entry_destroy
	MOVQ 64(SP), AX
	TESTQ AX, AX
	JZ entry_destroy
	MOVQ AX, 0(R12)
entry_destroy:
	LEAQ 0(SP), DI
	CALL x_pthread_attr_destroy(SB)

entry_setg:
	MOVQ R12, DI
	MOVQ savedSetg<>(SB), AX
	CALL AX
	CALL R13

	XORQ AX, AX
	ADDQ $96, SP
	POPQ R13
	POPQ R12
	POPQ BP
	RET

TEXT ·cgoNotifyInitDoneSym(SB), NOSPLIT|NOFRAME, $0
	RET

TEXT ·cgoSetenvSym(SB), NOSPLIT|NOFRAME, $0
	SUBQ $8, SP
	MOVQ 8(DI), SI
	MOVQ 0(DI), DI
	MOVL $1, DX
	CALL x_setenv(SB)
	ADDQ $8, SP
	RET

TEXT ·cgoUnsetenvSym(SB), NOSPLIT|NOFRAME, $0
	SUBQ $8, SP
	MOVQ 0(DI), DI
	CALL x_unsetenv(SB)
	ADDQ $8, SP
	RET
