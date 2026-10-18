package autograd

import (
	"fmt"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func BenchmarkCPUUnary(b *testing.B) {
	for _, n := range []int{256, 4096 * 1024} {
		b.Run(fmt.Sprintf("gelu/%d", n), func(b *testing.B) {
			x, _ := Zeros([]int{n}, tensor.CPU, true)
			defer x.Close()
			g := make([]float32, n)
			for i := range x.Data {
				x.Data[i] = float32(i%97-48) / 16
				g[i] = float32(i%13-6) / 8
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				y := GELU(x, false)
				y.backward(g)
				y.ReleaseGraph()
				x.ZeroGrad()
			}
		})
	}
}

// Measure the operation and its VJP without including a separate loss reduction.
func BenchmarkCPUSoftmax(b *testing.B) {
	for _, tc := range []struct {
		name  string
		shape []int
		axis  int
	}{
		{"small", []int{8, 32}, -1},
		{"attention", []int{16, 4, 256, 256}, -1},
		{"strided", []int{16, 256, 64}, 1},
	} {
		for _, logmode := range []bool{false, true} {
			name := "softmax/"
			if logmode {
				name = "log_softmax/"
			}
			b.Run(name+tc.name, func(b *testing.B) {
				x, _ := Zeros(tc.shape, tensor.CPU, true)
				defer x.Close()
				for i := range x.Data {
					x.Data[i] = float32(i%97-48) / 16
				}
				g := make([]float32, x.Numel())
				for i := range g {
					g[i] = float32(i%13-6) / 8
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					y := softmax(x, tc.axis, logmode)
					y.backward(g)
					y.ReleaseGraph()
					x.ZeroGrad()
				}
			})
		}
	}
}

func BenchmarkCPUMatMul(b *testing.B) {
	for _, tc := range []struct {
		name string
		a, b []int
	}{
		{"matrix", []int{512, 256}, []int{256, 512}},
		{"linear", []int{4, 128, 256}, []int{256, 512}},
		{"batched", []int{8, 64, 64}, []int{8, 64, 64}},
	} {
		for _, training := range []bool{false, true} {
			mode := "/forward"
			if training {
				mode = "/weight_grad"
			}
			b.Run(tc.name+mode, func(b *testing.B) {
				x, _ := Zeros(tc.a, tensor.CPU, false)
				w, _ := Zeros(tc.b, tensor.CPU, training)
				defer x.Close()
				defer w.Close()
				for i := range x.Data {
					x.Data[i] = float32(i%17-8) / 16
				}
				for i := range w.Data {
					w.Data[i] = float32(i%13-6) / 16
				}
				shape := append(bshape(tc.a[:len(tc.a)-2], tc.b[:len(tc.b)-2]), tc.a[len(tc.a)-2], tc.b[len(tc.b)-1])
				g := make([]float32, numel(shape))
				for i := range g {
					g[i] = 1
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					y := MatMul(x, w)
					if training {
						y.backward(g)
					}
					y.ReleaseGraph()
					w.ZeroGrad()
				}
			})
		}
	}
}
