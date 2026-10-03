package tensor

// The assembly microkernel is selected only when both the CPU and OS support
// AVX state, and the CPU exposes AVX2 and FMA. All other CPUs use kernel4x4.
func hasAVX2FMA() bool
func kernel4x4AVX(c, a, b *float32, stride, k int, add bool)
func kernel4x8AVX(c, a, b *float32, stride, k int, add bool)
func kernel8x8AVX(c, a, b *float32, stride, k int, add bool)

var useAVX2FMA = hasAVX2FMA()
