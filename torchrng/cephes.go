package torchrng

import "math"

// This file ports the float32 transcendental approximations that PyTorch's
// AVX2 kernels use for torch.randn, from aten/src/ATen/native/cpu/avx_mathfun.h.
// They are cephes minimax polynomials, not libm: they differ from a
// correctly-rounded logf/sinf/cosf by about one ulp, and that difference is
// visible in the sampled values. Every operation below is a single float32
// rounding, matching the corresponding _mm256_* intrinsic.

const (
	logP0 = 7.0376836292e-2
	logP1 = -1.1514610310e-1
	logP2 = 1.1676998740e-1
	logP3 = -1.2420140846e-1
	logP4 = 1.4249322787e-1
	logP5 = -1.6668057665e-1
	logP6 = 2.0000714765e-1
	logP7 = -2.4999993993e-1
	logP8 = 3.3333331174e-1
	logQ1 = -2.12194440e-4
	logQ2 = 0.693359375

	sqrthf = 0.707106781186547524

	dp1  = -0.78515625
	dp2  = -2.4187564849853515625e-4
	dp3  = -3.77489497744594108e-8
	fopi = 1.27323954473516

	sincof0 = -1.9515295891e-4
	sincof1 = 8.3321608736e-3
	sincof2 = -1.6666654611e-1

	coscof0 = 2.443315711809948e-005
	coscof1 = -1.388731625493765e-003
	coscof2 = 4.166664568298827e-002
)

const (
	minNormPos = 0x1p-126 // _ps256_min_norm_pos
	invSignBit = 0x7fffffff
	signBit    = 0x80000000
)

// log256 is log256_ps. Non-positive input returns the same negative NaN the
// intrinsic produces by OR-ing an all-ones mask into the result.
func log256(x float32) float32 {
	if x <= 0 {
		return math.Float32frombits(0xffffffff)
	}
	if x < minNormPos {
		x = minNormPos
	}
	bits := math.Float32bits(x)
	exponent := int32(uint32(bits) >> 23)
	// Keep the fraction and restore the exponent 0.5, so the mantissa lies in
	// [0.5, 1) regardless of the original exponent.
	bits &^= 0x7f800000
	bits |= math.Float32bits(0.5)
	x = math.Float32frombits(bits)

	exponent -= 0x7f
	e := float32(exponent) + 1

	// If the mantissa is below 1/sqrt(2), fold the leading one back in and
	// subtract one from the exponent.
	low := x < sqrthf
	var kept float32
	if low {
		kept = x
	}
	x = x - 1
	if low {
		e -= 1
	}
	x += kept

	z := x * x
	y := float32(logP0)
	y = y*x + logP1
	y = y*x + logP2
	y = y*x + logP3
	y = y*x + logP4
	y = y*x + logP5
	y = y*x + logP6
	y = y*x + logP7
	y = y*x + logP8
	y = y * x
	y = y * z
	y += e * logQ1
	y -= z * 0.5
	x += y
	x += e * logQ2
	return x
}

// sincos256 is sincos256_ps for a theta in [0, 2*pi), which is the only range
// Box-Muller produces.
func sincos256(theta float32) (sin, cos float32) {
	original := math.Float32bits(theta)
	absolute := math.Float32frombits(original & invSignBit)
	signSin := math.Float32frombits(original & signBit)

	quadrant := float32(int32(float64(absolute) * fopi))
	// The reference stores the truncated value, then rounds it up to the next
	// even integer with (j + 1) & ~1.
	truncated := int32(quadrant)
	even := (truncated + 1) &^ 1
	quadrant = float32(even)

	swapSin := math.Float32frombits(uint32((int32(even) & 4) << 29))
	useSinPoly := int32(even)&2 == 0

	signSin = xorSign(signSin, swapSin)

	r := absolute + quadrant*dp1
	r += quadrant * dp2
	r += quadrant * dp3

	// The cosine sign comes from (quadrant - 2), with the low bits of two pi
	// cleared, not from the quadrant itself.
	signCos := math.Float32frombits(uint32((^(even - 2) & 4) << 29))

	z := r * r
	c := float32(coscof0)
	c = c*z + coscof1
	c = c*z + coscof2
	c = c * z
	c = c * z
	c -= z * 0.5
	c += 1

	s := float32(sincof0)
	s = s*z + sincof1
	s = s*z + sincof2
	s = s * z
	s = s * r
	s = s + r

	// Even quadrants select the sine polynomial for sine and the cosine
	// polynomial for cosine; odd quadrants swap them. The reference stores the
	// quadrant rounded up to an even integer, so the test is on even&2.
	if !useSinPoly {
		s, c = c, s
	}

	return xorSign(s, signSin), xorSign(c, signCos)
}

func xorSign(value, sign float32) float32 {
	return math.Float32frombits(math.Float32bits(value) ^ math.Float32bits(sign))
}

// log1p32 is log1p(-x) evaluated in float32, for the serial fallback path.
func log1p32(x float32) float32 { return float32(math.Log1p(float64(x))) }

// normalFill16Vector is NormalFill16<float, true>: the AVX2 specialization.
func normalFill16Vector(d []float32, mean, std float32) {
	const twoPi = float32(2 * math.Pi) // _mm256_set1_ps(2.0f * c10::pi<double>)
	for j := 0; j < 8; j++ {
		u1 := 1 - d[j]
		u2 := d[j+8]
		radius := float32(math.Sqrt(float64(-2 * log256(u1))))
		theta := twoPi * u2
		sin, cos := sincos256(theta)
		d[j] = fma32(radius*cos, std, mean)
		d[j+8] = fma32(radius*sin, std, mean)
	}
}

// normalFill16Scalar is NormalFill16<double>, the non-vectorized template.
func normalFill16Scalar(d []float64, mean, std float64) {
	// The reference writes 2.0f * c10::pi<double>: the float literal is promoted
	// to double before the multiply, so two pi keeps full float64 precision and
	// is not rounded to float32 like the AVX2 float path's constant.
	twoPi := 2 * math.Pi
	for j := 0; j < 8; j++ {
		u1 := 1 - d[j]
		u2 := d[j+8]
		radius := math.Sqrt(-2 * math.Log(u1))
		theta := twoPi * u2
		sin, cos := math.Sincos(theta)
		d[j] = math.FMA(radius*cos, std, mean)
		d[j+8] = math.FMA(radius*sin, std, mean)
	}
}

// fma32 evaluates a*b+c with a single rounding, matching _mm256_fmadd_ps. The
// float64 intermediate makes the transform exact: the product of two float32
// values fits in 53 bits, and double rounding from float64 to float32 is
// innocuous because float64 carries more than 2*24+2 bits.
func fma32(a, b, c float32) float32 {
	return float32(float64(a)*float64(b) + float64(c))
}
