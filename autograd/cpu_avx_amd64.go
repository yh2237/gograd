//go:build amd64

package autograd

func hasAVX2FMA() bool
func expSubMaxAVX(dst, src *float32, n int, maxVal float32) float32
func geluFwdAVX(dst, src *float32, n int)
func geluBwdAVX(dst, src, grad *float32, n int)

var useCPUAVX2 = hasAVX2FMA()
