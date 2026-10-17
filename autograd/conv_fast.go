package autograd

import "github.com/yh2237/gograd/tensor"

// Conv1dGEMM lowers odd-kernel, same-padded convolution to matrix products.
func Conv1dGEMM(x, w, b *Tensor, dilation int) *Tensor {
	useCUDA := dispatchBackend("conv1d_gemm", x.Device)
	return conv1dGEMM(x, w, b, 1, 0, dilation, true, useCUDA)
}

// Conv1dGEMMStride applies a time-major 1D convolution with explicit stride,
// symmetric padding and dilation. Weights have shape [out, in, kernel].
func Conv1dGEMMStride(x, w, b *Tensor, stride, padding, dilation int) *Tensor {
	useCUDA := dispatchBackend("conv1d_gemm_stride", x.Device)
	return conv1dGEMM(x, w, b, stride, padding, dilation, false, useCUDA)
}

func conv1dGEMM(x, w, b *Tensor, stride, padding, dilation int, legacy, useCUDA bool) *Tensor {
	same(x, w, b)
	x, w, b = x.Contiguous(), w.Contiguous(), b.Contiguous()
	s := x.Shape
	if len(s) != 3 || len(w.Shape) != 3 || w.Shape[1] != s[2] || b.Numel() != w.Shape[0] || w.Shape[2] < 1 || stride < 1 || padding < 0 || dilation < 1 || (legacy && w.Shape[2]%2 != 1) {
		panic("autograd: conv1d shape")
	}
	if !legacy {
		return conv1dGEMMStrided(x, w, b, stride, padding, dilation, useCUDA)
	}
	if useCUDA {
		return gpuConv1d(x, w, b, dilation)
	}
	bs, t, ci, co, k := s[0], s[1], s[2], w.Shape[0], w.Shape[2]
	rows, depth := bs*t, ci*k
	cols := cpuAlloc(rows * depth)
	for n := 0; n < bs; n++ {
		for at := 0; at < t; at++ {
			for i := 0; i < ci; i++ {
				for j := 0; j < k; j++ {
					src := at + (j-k/2)*dilation
					if src >= 0 && src < t {
						cols[((n*t+at)*ci+i)*k+j] = x.Data[(n*t+src)*ci+i]
					}
				}
			}
		}
	}
	v := cpuAlloc(rows * co)
	tensor.SGEMMOp(v, cols, w.Data, rows, co, depth, false, true)
	for r := 0; r < rows; r++ {
		for o := 0; o < co; o++ {
			v[r*co+o] += b.Data[o]
		}
	}
	r := result(v, []int{bs, t, co}, []*Tensor{x, w, b}, func(g []float32) {
		db := cpuAlloc(co)
		for r := 0; r < rows; r++ {
			for o := 0; o < co; o++ {
				db[o] += g[r*co+o]
			}
		}
		dwT := cpuAlloc(depth * co)
		tensor.SGEMMOp(dwT, cols, g, depth, co, rows, true, false)
		dw := cpuAlloc(len(w.Data))
		for o := 0; o < co; o++ {
			for j := 0; j < depth; j++ {
				dw[o*depth+j] = dwT[j*co+o]
			}
		}
		dcols := cpuAlloc(len(cols))
		tensor.SGEMM(dcols, g, w.Data, rows, depth, co)
		dx := cpuAlloc(len(x.Data))
		for n := 0; n < bs; n++ {
			for at := 0; at < t; at++ {
				for i := 0; i < ci; i++ {
					for j := 0; j < k; j++ {
						src := at + (j-k/2)*dilation
						if src >= 0 && src < t {
							dx[(n*t+src)*ci+i] += dcols[((n*t+at)*ci+i)*k+j]
						}
					}
				}
			}
		}
		x.addGrad(dx)
		w.addGrad(dw)
		b.addGrad(db)
		cpuRelease(db)
		cpuRelease(dwT)
		cpuRelease(dw)
		cpuRelease(dcols)
		cpuRelease(dx)
	})
	if !r.RequiresGrad {
		cpuRelease(cols)
	} else {
		r.auxCPU = [][]float32{cols}
	}
	return r
}
