package autograd

import (
	"fmt"

	"github.com/yh2237/gograd/tensor"
)

// Conv2dOptions specifies a zero-padded cross-correlation. Pairs are
// [height,width]; Padding is applied symmetrically on each axis. An all-zero
// Stride or Dilation defaults to [1,1], and Groups defaults to 1.
type Conv2dOptions struct {
	Stride, Padding, Dilation [2]int
	Groups                    int
}

func (o Conv2dOptions) normalized() (Conv2dOptions, error) {
	if o.Stride == [2]int{} {
		o.Stride = [2]int{1, 1}
	}
	if o.Dilation == [2]int{} {
		o.Dilation = [2]int{1, 1}
	}
	if o.Groups == 0 {
		o.Groups = 1
	}
	const maxDim = 1<<31 - 1
	for i := 0; i < 2; i++ {
		if o.Stride[i] < 1 || o.Stride[i] > maxDim || o.Dilation[i] < 1 || o.Dilation[i] > maxDim || o.Padding[i] < 0 || o.Padding[i] > maxDim {
			return o, fmt.Errorf("autograd: invalid Conv2d stride, padding or dilation")
		}
	}
	if o.Groups < 1 || o.Groups > maxDim {
		return o, fmt.Errorf("autograd: invalid Conv2d groups")
	}
	return o, nil
}

// Conv2d uses NCHW input and PyTorch [out,in/groups,kH,kW] weights. Bias may
// be nil. Both devices use tiled im2col + GEMM; backward recomputes each tile
// rather than retaining a full lowered input. Even and rectangular kernels,
// strided/dilated convolution, groups and depthwise multipliers are supported.
func Conv2d(x, w, b *Tensor, options Conv2dOptions) *Tensor {
	if x == nil || w == nil {
		panic("autograd: Conv2d requires input and weight")
	}
	useCUDA := dispatchBackend("conv2d", x.Device)
	same(x, w, b)
	spec, err := conv2dShape(x.Shape, w.Shape, options)
	if err != nil {
		panic(err)
	}
	if b != nil && (len(b.Shape) != 1 || b.Shape[0] != spec.co) {
		panic("autograd: Conv2d bias must have shape [out_channels]")
	}
	x, w = x.Contiguous(), w.Contiguous()
	parents := []*Tensor{x, w}
	if b != nil {
		b = b.Contiguous()
		parents = append(parents, b)
	}
	if useCUDA {
		return gpuConv2d(x, w, b, spec, parents)
	}
	return cpuConv2d(x, w, b, spec, parents)
}

type conv2dSpec struct {
	batch, ci, h, width, co, kh, kw, oh, ow int
	options                                 Conv2dOptions
}

// Reject overflowing dimensions before allocating or passing CUDA int indices.
// The same limits apply on both devices so backend selection is transparent.
func conv2dProduct(shape ...int) (int, error) {
	const limit = 1<<31 - 1
	n := 1
	for _, d := range shape {
		if d < 1 || d > limit || n > limit/d {
			return 0, fmt.Errorf("autograd: invalid or oversized Conv2d dimensions %v", shape)
		}
		n *= d
	}
	return n, nil
}

func conv2dShape(xs, ws []int, o Conv2dOptions) (conv2dSpec, error) {
	s := conv2dSpec{}
	var err error
	if s.options, err = o.normalized(); err != nil {
		return s, err
	}
	if len(xs) != 4 || len(ws) != 4 {
		return s, fmt.Errorf("autograd: Conv2d expects rank-four input and weight")
	}
	if _, err = conv2dProduct(xs...); err != nil {
		return s, err
	}
	if _, err = conv2dProduct(ws...); err != nil {
		return s, err
	}
	s.batch, s.ci, s.h, s.width = xs[0], xs[1], xs[2], xs[3]
	s.co, s.kh, s.kw = ws[0], ws[2], ws[3]
	groups := s.options.Groups
	if s.ci%groups != 0 || s.co%groups != 0 || ws[1] != s.ci/groups {
		return s, fmt.Errorf("autograd: Conv2d channels/weight do not match groups")
	}
	for axis, input := range []int{s.h, s.width} {
		kernel := ws[axis+2]
		numerator := int64(input) + 2*int64(s.options.Padding[axis]) - int64(s.options.Dilation[axis])*int64(kernel-1) - 1
		if numerator < 0 {
			return s, fmt.Errorf("autograd: Conv2d effective kernel exceeds padded input")
		}
		dim := numerator/int64(s.options.Stride[axis]) + 1
		if dim > 1<<31-1 {
			return s, fmt.Errorf("autograd: oversized Conv2d output")
		}
		if axis == 0 {
			s.oh = int(dim)
		} else {
			s.ow = int(dim)
		}
	}
	_, err = conv2dProduct(s.outputShape()...)
	return s, err
}

func (s conv2dSpec) outputShape() []int { return []int{s.batch, s.co, s.oh, s.ow} }
func (s conv2dSpec) depth() int         { return s.ci / s.options.Groups * s.kh * s.kw }
func (s conv2dSpec) outChannels() int   { return s.co / s.options.Groups }

// At most 4 MiB for the pair of tile matrices, apart from a single column
// when the kernel itself exceeds that budget. Backward uses a second pair.
func (s conv2dSpec) tileColumns() int {
	return min(s.oh*s.ow, max(1, min(512, (4<<20)/4/(s.depth()+s.outChannels()))))
}

// Lower a single batch/group, depth-major so output is already in NCHW order.
func (s conv2dSpec) columns(dst, input []float32, batch, group, start, count int) {
	ci := s.ci / s.options.Groups
	for d := 0; d < s.depth(); d++ {
		channel, tap := d/(s.kh*s.kw), d%(s.kh*s.kw)
		ky, kx := tap/s.kw, tap%s.kw
		base := ((batch*s.ci + group*ci + channel) * s.h) * s.width
		for j := 0; j < count; j++ {
			p := start + j
			y := (p/s.ow)*s.options.Stride[0] - s.options.Padding[0] + ky*s.options.Dilation[0]
			x := (p%s.ow)*s.options.Stride[1] - s.options.Padding[1] + kx*s.options.Dilation[1]
			v := float32(0)
			if y >= 0 && y < s.h && x >= 0 && x < s.width {
				v = input[base+y*s.width+x]
			}
			dst[d*count+j] = v
		}
	}
}

func (s conv2dSpec) scatterColumns(dx, cols []float32, batch, group, start, count int) {
	ci := s.ci / s.options.Groups
	for d := 0; d < s.depth(); d++ {
		channel, tap := d/(s.kh*s.kw), d%(s.kh*s.kw)
		ky, kx := tap/s.kw, tap%s.kw
		base := ((batch*s.ci + group*ci + channel) * s.h) * s.width
		for j := 0; j < count; j++ {
			p := start + j
			y := (p/s.ow)*s.options.Stride[0] - s.options.Padding[0] + ky*s.options.Dilation[0]
			x := (p%s.ow)*s.options.Stride[1] - s.options.Padding[1] + kx*s.options.Dilation[1]
			if y >= 0 && y < s.h && x >= 0 && x < s.width {
				dx[base+y*s.width+x] += cols[d*count+j]
			}
		}
	}
}

func cpuConv2d(x, w, b *Tensor, s conv2dSpec, parents []*Tensor) *Tensor {
	depth, co, spatial, tile := s.depth(), s.outChannels(), s.oh*s.ow, s.tileColumns()
	cols, values := cpuAlloc(depth*tile), cpuAlloc(co*tile)
	defer cpuRelease(cols)
	defer cpuRelease(values)
	out := cpuAlloc(numel(s.outputShape()))
	for n := 0; n < s.batch; n++ {
		for group := 0; group < s.options.Groups; group++ {
			weight := w.Data[group*co*depth : (group+1)*co*depth]
			for start := 0; start < spatial; start += tile {
				count := min(tile, spatial-start)
				s.columns(cols[:depth*count], x.Data, n, group, start, count)
				tensor.SGEMM(values[:co*count], weight, cols[:depth*count], co, count, depth)
				for o := 0; o < co; o++ {
					channel := group*co + o
					base := (n*s.co+channel)*spatial + start
					for j := 0; j < count; j++ {
						v := values[o*count+j]
						if b != nil {
							v += b.Data[channel]
						}
						out[base+j] = v
					}
				}
			}
		}
	}
	return result(out, s.outputShape(), parents, func(g []float32) {
		var dx, dw, db []float32
		if x.RequiresGrad {
			dx = cpuAlloc(x.Numel())
			defer cpuRelease(dx)
		}
		if w.RequiresGrad {
			dw = cpuAlloc(w.Numel())
			defer cpuRelease(dw)
		}
		if b != nil && b.RequiresGrad {
			db = cpuAlloc(s.co)
			defer cpuRelease(db)
		}
		if dx != nil || dw != nil {
			cols, upstream := cpuAlloc(depth*tile), cpuAlloc(co*tile)
			defer cpuRelease(cols)
			defer cpuRelease(upstream)
			var partial []float32
			if dw != nil {
				partial = cpuAlloc(co * depth)
				defer cpuRelease(partial)
			}
			for n := 0; n < s.batch; n++ {
				for group := 0; group < s.options.Groups; group++ {
					offset := group * co * depth
					for start := 0; start < spatial; start += tile {
						count := min(tile, spatial-start)
						for o := 0; o < co; o++ {
							base := (n*s.co+group*co+o)*spatial + start
							copy(upstream[o*count:(o+1)*count], g[base:base+count])
						}
						if dw != nil {
							s.columns(cols[:depth*count], x.Data, n, group, start, count)
							tensor.SGEMMOp(partial, upstream[:co*count], cols[:depth*count], co, depth, count, false, true)
							for i, v := range partial {
								dw[offset+i] += v
							}
						}
						if dx != nil {
							tensor.SGEMMOp(cols[:depth*count], w.Data[offset:offset+co*depth], upstream[:co*count], depth, count, co, true, false)
							s.scatterColumns(dx, cols[:depth*count], n, group, start, count)
						}
					}
				}
			}
		}
		if db != nil {
			for n := 0; n < s.batch; n++ {
				for o := 0; o < s.co; o++ {
					for _, v := range g[(n*s.co+o)*spatial : (n*s.co+o+1)*spatial] {
						db[o] += v
					}
				}
			}
			b.addGrad(db)
		}
		if dx != nil {
			x.addGrad(dx)
		}
		if dw != nil {
			w.addGrad(dw)
		}
	})
}
