package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"github.com/yh2237/gograd/tensor"
)

func conv1dGEMMStrided(x, w, b *Tensor, stride, padding, dilation int, useCUDA bool) *Tensor {
	bs, length, ci, co, k := x.Shape[0], x.Shape[1], x.Shape[2], w.Shape[0], w.Shape[2]
	span := dilation*(k-1) + 1
	if length+2*padding < span {
		panic("autograd: conv1d shape")
	}
	outLength := (length+2*padding-span)/stride + 1
	rows, depth := bs*outLength, ci*k
	if useCUDA {
		return gpuConv1dStride(x, w, b, bs, length, ci, co, k, outLength, rows, depth, stride, padding, dilation)
	}
	cols := cpuAlloc(rows * depth)
	for n := 0; n < bs; n++ {
		for at := 0; at < outLength; at++ {
			for c := 0; c < ci; c++ {
				for tap := 0; tap < k; tap++ {
					src := at*stride - padding + tap*dilation
					if src >= 0 && src < length {
						cols[((n*outLength+at)*ci+c)*k+tap] = x.Data[(n*length+src)*ci+c]
					}
				}
			}
		}
	}
	v := cpuAlloc(rows * co)
	tensor.SGEMMOp(v, cols, w.Data, rows, co, depth, false, true)
	for row := 0; row < rows; row++ {
		for o := 0; o < co; o++ {
			v[row*co+o] += b.Data[o]
		}
	}
	r := result(v, []int{bs, outLength, co}, []*Tensor{x, w, b}, func(g []float32) {
		db := cpuAlloc(co)
		for row := 0; row < rows; row++ {
			for o := 0; o < co; o++ {
				db[o] += g[row*co+o]
			}
		}
		dw := cpuAlloc(len(w.Data))
		tensor.SGEMMOp(dw, g, cols, co, depth, rows, true, false)
		dcols := cpuAlloc(len(cols))
		tensor.SGEMM(dcols, g, w.Data, rows, depth, co)
		dx := cpuAlloc(len(x.Data))
		for n := 0; n < bs; n++ {
			for at := 0; at < outLength; at++ {
				for c := 0; c < ci; c++ {
					for tap := 0; tap < k; tap++ {
						src := at*stride - padding + tap*dilation
						if src >= 0 && src < length {
							dx[(n*length+src)*ci+c] += dcols[((n*outLength+at)*ci+c)*k+tap]
						}
					}
				}
			}
		}
		x.addGrad(dx)
		w.addGrad(dw)
		b.addGrad(db)
		cpuRelease(db)
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

func gpuConv1dStride(x, w, b *Tensor, bs, length, ci, co, k, outLength, rows, depth, stride, padding, dilation int) *Tensor {
	options := executionOptions(x, w, b)
	cols := mustAlloc(rows * depth)
	out := mustAlloc(rows * co)
	timedGPU("conv_im2col_stride", func() error {
		return kernels.Im2colStride(x.buf, cols, bs, ci, length, k, outLength, stride, padding, dilation)
	})
	var cols16 *cuda.Buffer
	var wp16 uintptr
	if options.BF16Autocast {
		cols16 = castBF16(cols.Pointer(), rows*depth)
		wp16 = ptr(w.ensureBF16())
	}
	timedGPU("conv_forward_sgemm", func() error {
		return options.gemmSinglePrepared(rows, co, depth, false, true, cols.Pointer(), ptr(cols16), w.buf.Pointer(), wp16, out.Pointer(), 0)
	})
	timedGPU("conv_bias", func() error { return kernels.AddBiasColumns(out, b.buf, rows, co) })
	r := resultGPU(out, []int{bs, outLength, co}, []*Tensor{x, w, b}, func(g *cuda.Buffer) {
		var g16 *cuda.Buffer
		if options.BF16Autocast && (w.RequiresGrad || x.RequiresGrad) {
			g16 = castBF16(g.Pointer(), g.Size()/4)
			defer g16.Free()
		}
		if dw := w.ensureGradGPU(); dw != nil {
			timedGPU("conv_dw_sgemm", func() error {
				return options.gemmSinglePrepared(co, depth, rows, true, false, g.Pointer(), ptr(g16), cols.Pointer(), ptr(cols16), dw.Pointer(), 1)
			})
		}
		if db := b.ensureGradGPU(); db != nil {
			timedGPU("conv_db", func() error { return kernels.ColumnSum(g, db, rows, co) })
		}
		if dx := x.ensureGradGPU(); dx != nil {
			dcols := mustAlloc(rows * depth)
			defer dcols.Free()
			timedGPU("conv_dx_sgemm", func() error {
				return options.gemmSinglePrepared(rows, depth, co, false, false, g.Pointer(), ptr(g16), w.buf.Pointer(), wp16, dcols.Pointer(), 0)
			})
			timedGPU("conv_col2im_stride", func() error {
				return kernels.Col2imStride(dcols, dx, bs, ci, length, k, outLength, stride, padding, dilation)
			})
		}
	})
	r.aux = []*cuda.Buffer{cols}
	if cols16 != nil {
		r.aux = append(r.aux, cols16)
	}
	return r
}
