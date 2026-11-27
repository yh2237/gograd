//go:build amd64

#include "textflag.h"

DATA fInvLn2<>(SB)/4, $0x3fb8aa3b    // 1.44269504
GLOBL fInvLn2<>(SB), RODATA|NOPTR, $4

DATA fLn2Hi<>(SB)/4, $0x3f317200     // 0.693145751953125
GLOBL fLn2Hi<>(SB), RODATA|NOPTR, $4

DATA fLn2Lo<>(SB)/4, $0x35bfbe8e     // 1.428606765330187e-06
GLOBL fLn2Lo<>(SB), RODATA|NOPTR, $4

DATA fExpMin<>(SB)/4, $0xc2aeac50    // -87.336548
GLOBL fExpMin<>(SB), RODATA|NOPTR, $4

DATA fOne<>(SB)/4, $0x3f800000       // 1.0
GLOBL fOne<>(SB), RODATA|NOPTR, $4

DATA fHalf<>(SB)/4, $0x3f000000      // 0.5
GLOBL fHalf<>(SB), RODATA|NOPTR, $4

DATA fNegHalf<>(SB)/4, $0xbf000000   // -0.5
GLOBL fNegHalf<>(SB), RODATA|NOPTR, $4

DATA fExpC3<>(SB)/4, $0x3e2aaaab     // 1/6
GLOBL fExpC3<>(SB), RODATA|NOPTR, $4

DATA fExpC4<>(SB)/4, $0x3d2aaaab     // 1/24
GLOBL fExpC4<>(SB), RODATA|NOPTR, $4

DATA fExpC5<>(SB)/4, $0x3c088889     // 1/120
GLOBL fExpC5<>(SB), RODATA|NOPTR, $4

DATA i127<>(SB)/4, $127
GLOBL i127<>(SB), RODATA|NOPTR, $4

DATA fAbsMask<>(SB)/4, $0x7fffffff
GLOBL fAbsMask<>(SB), RODATA|NOPTR, $4

DATA fSignMask<>(SB)/4, $0x80000000
GLOBL fSignMask<>(SB), RODATA|NOPTR, $4

DATA fInvSqrt2<>(SB)/4, $0x3f3504f3  // 0.70710678
GLOBL fInvSqrt2<>(SB), RODATA|NOPTR, $4

DATA fInvSqrt2Pi<>(SB)/4, $0x3ecc422a // 0.39894228
GLOBL fInvSqrt2Pi<>(SB), RODATA|NOPTR, $4

DATA fTen<>(SB)/4, $0x41200000       // 10.0
GLOBL fTen<>(SB), RODATA|NOPTR, $4

DATA fNegTen<>(SB)/4, $0xc1200000    // -10.0
GLOBL fNegTen<>(SB), RODATA|NOPTR, $4

DATA fErfP<>(SB)/4, $0x3ea7ba05      // 0.3275911
GLOBL fErfP<>(SB), RODATA|NOPTR, $4

DATA fErfA1<>(SB)/4, $0x3e827906     // 0.254829592
GLOBL fErfA1<>(SB), RODATA|NOPTR, $4

DATA fErfA2<>(SB)/4, $0xbe91a98e     // -0.284496736
GLOBL fErfA2<>(SB), RODATA|NOPTR, $4

DATA fErfA3<>(SB)/4, $0x3fb5f0e3     // 1.421413741
GLOBL fErfA3<>(SB), RODATA|NOPTR, $4

DATA fErfA4<>(SB)/4, $0xbfba00e3     // -1.453152027
GLOBL fErfA4<>(SB), RODATA|NOPTR, $4

DATA fErfA5<>(SB)/4, $0x3f87dc22     // 1.061405429
GLOBL fErfA5<>(SB), RODATA|NOPTR, $4

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

// expSubMaxAVX computes dst[i] = exp(src[i] - maxVal) for 8*floor(n/8) elements
// and returns their partial sum. Caller handles any tail elements (< 8).
TEXT ·expSubMaxAVX(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS maxVal+24(FP), Y15

	VXORPS Y0, Y0, Y0 // accumulator
	SHRQ $3, CX
	JZ exp_done

	VBROADCASTSS fInvLn2<>(SB), Y8
	VBROADCASTSS fLn2Hi<>(SB), Y9
	VBROADCASTSS fLn2Lo<>(SB), Y10
	VBROADCASTSS fExpMin<>(SB), Y11
	VBROADCASTSS fOne<>(SB), Y12
	VBROADCASTSS fHalf<>(SB), Y13
	VPBROADCASTD i127<>(SB), Y14

exp_loop:
	VMOVUPS (SI), Y1
	VSUBPS Y15, Y1, Y1                // x = src - maxVal (<= 0)
	VCMPPS $13, Y11, Y1, Y7           // mask: x >= -87.336548
	VMAXPS Y1, Y11, Y1                // clamp x (propagating NaN)

	VMULPS Y8, Y1, Y2                 // scaled = x * invLn2
	VROUNDPS $0, Y2, Y2               // kf = round(scaled)
	VCVTPS2DQ Y2, Y6                  // k = int32(kf)

	VFNMADD231PS Y9, Y2, Y1           // r = x - kf*ln2Hi
	VFNMADD231PS Y10, Y2, Y1          // r -= kf*ln2Lo

	// p = 1 + r*(1 + r*(0.5 + r*(c3 + r*(c4 + r*c5))))
	VBROADCASTSS fExpC5<>(SB), Y3
	VBROADCASTSS fExpC4<>(SB), Y4
	VBROADCASTSS fExpC3<>(SB), Y5
	VFMADD213PS Y4, Y1, Y3            // c5*r + c4
	VFMADD213PS Y5, Y1, Y3            // *r + c3
	VFMADD213PS Y13, Y1, Y3           // *r + 0.5
	VFMADD213PS Y12, Y1, Y3           // *r + 1.0
	VFMADD213PS Y12, Y1, Y3           // *r + 1.0

	VPADDD Y14, Y6, Y6                // k + 127
	VPSLLD $23, Y6, Y6                // 2^k
	VMULPS Y6, Y3, Y3                 // p * 2^k
	VANDPS Y7, Y3, Y3                 // zero underflowed lanes

	VMOVUPS Y3, (DI)
	VADDPS Y3, Y0, Y0

	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ exp_loop

exp_done:
	// Horizontal sum of Y0 -> X0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+32(FP)
	VZEROUPPER
	RET

// geluFwdAVX computes exact-form GELU (via A&S erf approximation) for 8*floor(n/8) elements.
TEXT ·geluFwdAVX(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $3, CX
	JZ geluf_done

	VBROADCASTSS fInvLn2<>(SB), Y10
	VBROADCASTSS fLn2Hi<>(SB), Y11
	VBROADCASTSS fLn2Lo<>(SB), Y12
	VBROADCASTSS fOne<>(SB), Y13
	VBROADCASTSS fHalf<>(SB), Y14
	VPBROADCASTD i127<>(SB), Y15

geluf_loop:
	VMOVUPS (SI), Y0                  // x
	// Clamp x to [-10, 10] for polynomial evaluation so -0.5*x^2 stays in [-50, 0].
	VBROADCASTSS fTen<>(SB), Y8
	VBROADCASTSS fNegTen<>(SB), Y9
	VMINPS Y0, Y8, Y1
	VMAXPS Y1, Y9, Y1                 // xc = clamp(x, -10, 10)

	// e = exp(-0.5 * xc^2)
	VMULPS Y1, Y1, Y2
	VBROADCASTSS fNegHalf<>(SB), Y3
	VMULPS Y3, Y2, Y2                 // z = -0.5 * xc^2 in [-50, 0]

	VMULPS Y10, Y2, Y3
	VROUNDPS $0, Y3, Y3               // kf
	VCVTPS2DQ Y3, Y6                  // k
	VFNMADD231PS Y11, Y3, Y2          // r
	VFNMADD231PS Y12, Y3, Y2

	VBROADCASTSS fExpC5<>(SB), Y4
	VBROADCASTSS fExpC4<>(SB), Y5
	VBROADCASTSS fExpC3<>(SB), Y7
	VFMADD213PS Y5, Y2, Y4
	VFMADD213PS Y7, Y2, Y4
	VFMADD213PS Y14, Y2, Y4
	VFMADD213PS Y13, Y2, Y4
	VFMADD213PS Y13, Y2, Y4
	VPADDD Y15, Y6, Y6
	VPSLLD $23, Y6, Y6
	VMULPS Y6, Y4, Y4                 // Y4 = e = exp(-0.5 * xc^2)

	// ax = |xc| * invSqrt2
	VBROADCASTSS fAbsMask<>(SB), Y5
	VANDPS Y5, Y1, Y2                 // |xc|
	VBROADCASTSS fInvSqrt2<>(SB), Y6
	VMULPS Y6, Y2, Y2                 // ax

	// t = 1 / (1 + p*ax)
	VBROADCASTSS fErfP<>(SB), Y3
	VFMADD213PS Y13, Y2, Y3           // 1 + p*ax
	VDIVPS Y3, Y13, Y3                // t

	// poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * e
	VBROADCASTSS fErfA5<>(SB), Y5
	VBROADCASTSS fErfA4<>(SB), Y6
	VBROADCASTSS fErfA3<>(SB), Y7
	VFMADD213PS Y6, Y3, Y5
	VFMADD213PS Y7, Y3, Y5
	VBROADCASTSS fErfA2<>(SB), Y6
	VBROADCASTSS fErfA1<>(SB), Y7
	VFMADD213PS Y6, Y3, Y5
	VFMADD213PS Y7, Y3, Y5
	VMULPS Y3, Y5, Y5
	VMULPS Y4, Y5, Y5
	VSUBPS Y5, Y13, Y5                // 1 - poly

	// Apply sign of xc -> erf(xc/sqrt2)
	VBROADCASTSS fSignMask<>(SB), Y6
	VANDPS Y6, Y1, Y6                 // sign bit of xc
	VXORPS Y6, Y5, Y5                 // erf

	// y = 0.5 * xc * (1 + erf)
	VADDPS Y13, Y5, Y5
	VMULPS Y14, Y5, Y5
	VMULPS Y1, Y5, Y5

	// Handle saturation: x > 10 -> x, x < -10 -> 0
	VCMPPS $14, Y8, Y0, Y6            // mask: x > 10
	VBLENDVPS Y6, Y0, Y5, Y5
	VCMPPS $1, Y9, Y0, Y7             // mask: x < -10
	VXORPS Y2, Y2, Y2
	VBLENDVPS Y7, Y2, Y5, Y5

	VMOVUPS Y5, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ geluf_loop

geluf_done:
	VZEROUPPER
	RET

// geluBwdAVX accumulates dst[i] += grad[i] * dGELU(src[i]) for 8*floor(n/8) elements.
TEXT ·geluBwdAVX(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ grad+16(FP), DX
	MOVQ n+24(FP), CX
	SHRQ $3, CX
	JZ gelub_done

	VBROADCASTSS fInvLn2<>(SB), Y10
	VBROADCASTSS fLn2Hi<>(SB), Y11
	VBROADCASTSS fLn2Lo<>(SB), Y12
	VBROADCASTSS fOne<>(SB), Y13
	VBROADCASTSS fHalf<>(SB), Y14
	VPBROADCASTD i127<>(SB), Y15

gelub_loop:
	VMOVUPS (SI), Y0                  // x
	VBROADCASTSS fTen<>(SB), Y8
	VBROADCASTSS fNegTen<>(SB), Y9
	VMINPS Y0, Y8, Y1
	VMAXPS Y1, Y9, Y1                 // xc

	// e = exp(-0.5 * xc^2)
	VMULPS Y1, Y1, Y2
	VBROADCASTSS fNegHalf<>(SB), Y3
	VMULPS Y3, Y2, Y2

	VMULPS Y10, Y2, Y3
	VROUNDPS $0, Y3, Y3
	VCVTPS2DQ Y3, Y6
	VFNMADD231PS Y11, Y3, Y2
	VFNMADD231PS Y12, Y3, Y2

	VBROADCASTSS fExpC5<>(SB), Y4
	VBROADCASTSS fExpC4<>(SB), Y5
	VBROADCASTSS fExpC3<>(SB), Y7
	VFMADD213PS Y5, Y2, Y4
	VFMADD213PS Y7, Y2, Y4
	VFMADD213PS Y14, Y2, Y4
	VFMADD213PS Y13, Y2, Y4
	VFMADD213PS Y13, Y2, Y4
	VPADDD Y15, Y6, Y6
	VPSLLD $23, Y6, Y6
	VMULPS Y6, Y4, Y4                 // Y4 = e = exp(-0.5 * xc^2)

	// ax = |xc| * invSqrt2
	VBROADCASTSS fAbsMask<>(SB), Y5
	VANDPS Y5, Y1, Y2
	VBROADCASTSS fInvSqrt2<>(SB), Y6
	VMULPS Y6, Y2, Y2

	// t = 1 / (1 + p*ax)
	VBROADCASTSS fErfP<>(SB), Y3
	VFMADD213PS Y13, Y2, Y3
	VDIVPS Y3, Y13, Y3

	VBROADCASTSS fErfA5<>(SB), Y5
	VBROADCASTSS fErfA4<>(SB), Y6
	VBROADCASTSS fErfA3<>(SB), Y7
	VFMADD213PS Y6, Y3, Y5
	VFMADD213PS Y7, Y3, Y5
	VBROADCASTSS fErfA2<>(SB), Y6
	VBROADCASTSS fErfA1<>(SB), Y7
	VFMADD213PS Y6, Y3, Y5
	VFMADD213PS Y7, Y3, Y5
	VMULPS Y3, Y5, Y5
	VMULPS Y4, Y5, Y5
	VSUBPS Y5, Y13, Y5

	VBROADCASTSS fSignMask<>(SB), Y6
	VANDPS Y6, Y1, Y6
	VXORPS Y6, Y5, Y5                 // erf

	// dgelu = 0.5*(1 + erf) + xc * e * invSqrt2Pi
	VADDPS Y13, Y5, Y5
	VMULPS Y14, Y5, Y5
	VBROADCASTSS fInvSqrt2Pi<>(SB), Y6
	VMULPS Y6, Y4, Y4
	VFMADD231PS Y4, Y1, Y5            // Y5 = dgelu

	// Saturation: x > 10 -> 1.0, x < -10 -> 0.0
	VCMPPS $14, Y8, Y0, Y6            // x > 10
	VBLENDVPS Y6, Y13, Y5, Y5
	VCMPPS $1, Y9, Y0, Y7             // x < -10
	VXORPS Y2, Y2, Y2
	VBLENDVPS Y7, Y2, Y5, Y5

	VMOVUPS (DX), Y3                  // grad
	VMOVUPS (DI), Y2                  // dst
	VFMADD231PS Y3, Y5, Y2            // dst += grad * dgelu
	VMOVUPS Y2, (DI)

	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	DECQ CX
	JNZ gelub_loop

gelub_done:
	VZEROUPPER
	RET
