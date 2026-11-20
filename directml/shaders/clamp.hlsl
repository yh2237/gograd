#define ClampRoot "UAV(u0), UAV(u1), RootConstants(num32BitConstants=4, b0)"

RWStructuredBuffer<uint> inputBits : register(u0);
RWStructuredBuffer<uint> outputBits : register(u1);

cbuffer ClampArgs : register(b0) {
    uint count;
    uint lowerBits;
    uint upperBits;
    uint rowStride;
};

// Integer comparisons preserve subnormals and put -0 before +0.
uint ordered(uint bits) {
    return bits ^ (asint(bits) < 0 ? 0xffffffffu : 0x80000000u);
}

[RootSignature(ClampRoot)]
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i = thread.x + thread.y * rowStride;
    if (i >= count) return;

    uint bits = inputBits[i];
    if ((bits & 0x7fffffffu) <= 0x7f800000u) {
        bits = ordered(bits) < ordered(lowerBits) ? lowerBits : bits;
        bits = ordered(bits) > ordered(upperBits) ? upperBits : bits;
    }
    outputBits[i] = bits;
}
