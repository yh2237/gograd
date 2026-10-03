package autograd

import "github.com/yh2237/gograd/tensor"

// Conv1dGEMM lowers convolution to contiguous matrix products. CUDA currently
// stages these matrices on the host; the device kernel migration is pending.
func Conv1dGEMM(x, w, b *Tensor, dilation int) *Tensor {
	same(x, w)
	same(x, b)
	s := x.Shape
	if len(s) != 3 || len(w.Shape) != 3 || w.Shape[1] != s[2] || len(b.Data) != w.Shape[0] || w.Shape[2]%2 != 1 || dilation < 1 {
		panic("autograd: conv1d shape")
	}
	bs, t, ci, co, k := s[0], s[1], s[2], w.Shape[0], w.Shape[2]
	rows, depth := bs*t, ci*k
	cols := make([]float32, rows*depth)
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
	v := make([]float32, rows*co)
	tensor.SGEMMOp(v, cols, w.Data, rows, co, depth, false, true)
	for r := 0; r < rows; r++ {
		for o := 0; o < co; o++ {
			v[r*co+o] += b.Data[o]
		}
	}
	return result(v, []int{bs, t, co}, []*Tensor{x, w, b}, func(g []float32) {
		db := make([]float32, co)
		for r := 0; r < rows; r++ {
			for o := 0; o < co; o++ {
				db[o] += g[r*co+o]
			}
		}
		dwT := make([]float32, depth*co)
		tensor.SGEMMOp(dwT, cols, g, depth, co, rows, true, false)
		dw := make([]float32, len(w.Data))
		for o := 0; o < co; o++ {
			for j := 0; j < depth; j++ {
				dw[o*depth+j] = dwT[j*co+o]
			}
		}
		dcols := make([]float32, len(cols))
		tensor.SGEMM(dcols, g, w.Data, rows, depth, co)
		dx := make([]float32, len(x.Data))
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
	})
}
