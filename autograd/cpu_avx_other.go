//go:build !amd64

package autograd

var useCPUAVX2 = false

func expSubMaxAVX(dst, src *float32, n int, maxVal float32) float32 {
	panic("autograd: AVX2 unavailable")
}
func geluFwdAVX(dst, src *float32, n int)       { panic("autograd: AVX2 unavailable") }
func geluBwdAVX(dst, src, grad *float32, n int) { panic("autograd: AVX2 unavailable") }
