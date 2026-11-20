#define UnaryRoot "UAV(u0), UAV(u1), RootConstants(num32BitConstants=6, b0)"

RWStructuredBuffer<uint> inputBits : register(u0);
RWStructuredBuffer<uint> outputBits : register(u1);

cbuffer UnaryArgs : register(b0) {
    uint count;
    uint parameter;
    uint extra;
    uint rowStride;
    uint powLower;
    uint powUpper;
};

static const uint signBit = 0x80000000u;
static const uint one = 0x3f800000u;
static const uint infinity = 0x7f800000u;
static const uint quietNaN = 0x7fc00000u;
static const uint smallestNormal = 0x00800000u;

// Integer comparisons preserve subnormals and put -0 before +0.
uint ordered(uint bits) {
    return bits ^ (asint(bits) < 0 ? 0xffffffffu : signBit);
}

[RootSignature(UnaryRoot)]
[numthreads(256, 1, 1)]
void clampMain(uint3 thread : SV_DispatchThreadID) {
    uint i = thread.x + thread.y * rowStride;
    if (i >= count) return;

    uint bits = inputBits[i];
    if ((bits & ~signBit) <= infinity) {
        bits = ordered(bits) < ordered(parameter) ? parameter : bits;
        bits = ordered(bits) > ordered(extra) ? extra : bits;
    }
    outputBits[i] = bits;
}

uint powMagnitude(uint magnitude, float exponent) {
    float x = asfloat(magnitude);
    float logMagnitude;
    if (magnitude < smallestNormal) {
        // Arithmetic flushes subnormals, so use the mantissa in units of 2^-149.
        logMagnitude = log2(float(magnitude)) - 149.0;
    } else if (abs(exponent) > 128.0 && x >= 0.625 && x <= 1.5) {
        // log2(x) = 2 * log2(e) * atanh((x - 1) / (x + 1)).
        float u = (x - 1.0) / (x + 1.0);
        float u2 = u * u;
        float series = 1.0 / 9.0 + u2 / 11.0;
        series = 1.0 / 7.0 + u2 * series;
        series = 1.0 / 5.0 + u2 * series;
        series = 1.0 / 3.0 + u2 * series;
        logMagnitude = (2.8853900817779268 * u) * (1.0 + u2 * series);
    } else {
        logMagnitude = log2(x);
    }
    float power = exponent * logMagnitude;
    // Subnormal values and the smallest normal exponent share units of 2^-149.
    bool scaleUp = power < -125.0;
    float result = exp2(scaleUp ? power + 149.0 : power);
    return scaleUp ? max(1u, uint(round(result))) : min(infinity - 1, asuint(result));
}

uint powBits(uint bits, uint exponentBits, uint exponentClass) {
    uint magnitude = bits & ~signBit;
    uint exponentMagnitude = exponentBits & ~signBit;
    bool inverse = (exponentBits & signBit) != 0;
    uint sign = exponentClass == 2 ? bits & signBit : 0;
    uint result = quietNaN;
    if (bits == one) {
        result = one;
    } else if (magnitude <= infinity && exponentMagnitude <= infinity) {
        if (exponentMagnitude == infinity) {
            result = (magnitude > one) != inverse ? infinity : 0;
            if (magnitude == one) result = one;
        } else if (magnitude == 0) {
            result = sign | (inverse ? infinity : 0);
        } else if (magnitude == infinity) {
            result = sign | (inverse ? 0 : infinity);
        } else if ((bits & signBit) == 0 || exponentClass != 0) {
            if (magnitude == one || exponentMagnitude < smallestNormal) {
                result = sign | one;
            } else if (magnitude < powLower) {
                result = sign | (inverse ? infinity : 0);
            } else if (magnitude >= powUpper) {
                result = sign | (inverse ? 0 : infinity);
            } else {
                result = sign | powMagnitude(magnitude, asfloat(exponentBits));
            }
        }
    }
    return result;
}

[RootSignature(UnaryRoot)]
[numthreads(256, 1, 1)]
void powMain(uint3 thread : SV_DispatchThreadID) {
    uint i = thread.x + thread.y * rowStride;
    if (i < count) outputBits[i] = powBits(inputBits[i], parameter, extra);
}
