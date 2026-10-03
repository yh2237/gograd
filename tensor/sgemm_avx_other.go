//go:build !amd64

package tensor

var useAVX2FMA = false

func kernel4x4AVX(c, a, b *float32, stride, k int, add bool) { panic("tensor: AVX2 unavailable") }
