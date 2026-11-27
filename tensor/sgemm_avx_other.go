//go:build !amd64

package tensor

var useAVX2FMA = false

func kernel4x8AVX(c, a, b *float32, stride, k int, add bool)  { panic("tensor: AVX2 unavailable") }
func kernel4x16AVX(c, a, b *float32, stride, k int, add bool) { panic("tensor: AVX2 unavailable") }
func kernel6x16AVX(c, a, b *float32, stride, k int, add bool) { panic("tensor: AVX2 unavailable") }
