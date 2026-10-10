package autograd

import (
	"fmt"
	"math"
)

// MaxPool2dOptions uses [height,width] pairs. KernelSize is required; zero
// Stride defaults to KernelSize and zero Dilation defaults to [1,1]. Padding
// is symmetric and must not exceed half the kernel size on either axis.
type MaxPool2dOptions struct {
	KernelSize, Stride, Padding, Dilation [2]int
	CeilMode                              bool
}

// AvgPool2dOptions uses [height,width] pairs. Zero Stride defaults to KernelSize.
// Padding counts in the divisor by default; ExcludePad excludes it instead.
// A positive DivisorOverride replaces either divisor. Ceil-mode windows omit
// positions beyond the requested padding even when ExcludePad is false.
type AvgPool2dOptions struct {
	KernelSize, Stride, Padding [2]int
	CeilMode, ExcludePad        bool
	DivisorOverride             int
}

// MaxPool2d pools NCHW input. Padding acts as negative infinity; equal finite
// maxima select the first row-major sample, while NaNs select the last NaN.
func MaxPool2d(x *Tensor, options MaxPool2dOptions) *Tensor {
	useCUDA := dispatchBackend("max_pool2d", x.Device)
	x.assertOpen()
	s, err := pool2dShape(x.Shape, options.KernelSize, options.Stride, options.Padding, options.Dilation, options.CeilMode)
	if err != nil {
		panic(err)
	}
	x = x.Contiguous()
	if useCUDA {
		return gpuPool2d(x, s, 0)
	}
	return cpuPool2d(x, s, 0)
}

// AvgPool2d pools NCHW input with zero padding and the requested divisor.
func AvgPool2d(x *Tensor, options AvgPool2dOptions) *Tensor {
	useCUDA := dispatchBackend("avg_pool2d", x.Device)
	x.assertOpen()
	s, err := pool2dShape(x.Shape, options.KernelSize, options.Stride, options.Padding, [2]int{1, 1}, options.CeilMode)
	if err != nil {
		panic(err)
	}
	if options.DivisorOverride < 0 || options.DivisorOverride > 1<<31-1 {
		panic("autograd: invalid AvgPool2d divisor override")
	}
	s.excludePad, s.divisor = options.ExcludePad, options.DivisorOverride
	x = x.Contiguous()
	if useCUDA {
		return gpuPool2d(x, s, 1)
	}
	return cpuPool2d(x, s, 1)
}

// AdaptiveAvgPool2d averages NCHW input into a positive [height,width] size.
// Non-divisible bins overlap; output dimensions may exceed the input dimensions.
func AdaptiveAvgPool2d(x *Tensor, outputSize [2]int) *Tensor {
	useCUDA := dispatchBackend("adaptive_avg_pool2d", x.Device)
	x.assertOpen()
	s, err := adaptivePool2dShape(x.Shape, outputSize)
	if err != nil {
		panic(err)
	}
	x = x.Contiguous()
	if useCUDA {
		return gpuPool2d(x, s, 2)
	}
	return cpuPool2d(x, s, 2)
}

// Pooling layers have no parameters or train/eval state and compose directly
// through TensorLayer/Sequential.
type MaxPool2dLayer struct{ Options MaxPool2dOptions }

func (l MaxPool2dLayer) Forward(x *Tensor) *Tensor { return MaxPool2d(x, l.Options) }

type AvgPool2dLayer struct{ Options AvgPool2dOptions }

func (l AvgPool2dLayer) Forward(x *Tensor) *Tensor { return AvgPool2d(x, l.Options) }

type AdaptiveAvgPool2dLayer struct{ OutputSize [2]int }

func (l AdaptiveAvgPool2dLayer) Forward(x *Tensor) *Tensor { return AdaptiveAvgPool2d(x, l.OutputSize) }

type pool2dSpec struct {
	batch, channels, h, w, oh, ow     int
	kernel, stride, padding, dilation [2]int
	excludePad                        bool
	divisor                           int
}

func pool2dElements(shape []int) error {
	const limit = 1<<31 - 1
	n := 1
	for _, d := range shape {
		if d < 1 || d > limit || n > limit/d {
			return fmt.Errorf("autograd: invalid or oversized pooling shape %v", shape)
		}
		n *= d
	}
	return nil
}

func adaptivePool2dShape(shape []int, output [2]int) (pool2dSpec, error) {
	s := pool2dSpec{}
	if len(shape) != 4 {
		return s, fmt.Errorf("autograd: pooling requires NCHW rank-four input")
	}
	if err := pool2dElements(shape); err != nil {
		return s, err
	}
	s.batch, s.channels, s.h, s.w = shape[0], shape[1], shape[2], shape[3]
	s.oh, s.ow = output[0], output[1]
	return s, pool2dElements(s.outputShape())
}

// Go truncates negative integer division toward zero; pooling requires floor.
func poolFloor(n, d int64) int64 {
	if n < 0 {
		return -((-n + d - 1) / d)
	}
	return n / d
}

func pool2dShape(shape []int, kernel, stride, padding, dilation [2]int, ceil bool) (pool2dSpec, error) {
	s, err := adaptivePool2dShape(shape, [2]int{1, 1})
	if err != nil {
		return s, err
	}
	if stride == [2]int{} {
		stride = kernel
	}
	if dilation == [2]int{} {
		dilation = [2]int{1, 1}
	}
	s.kernel, s.stride, s.padding, s.dilation = kernel, stride, padding, dilation
	output := [2]int{}
	for axis, input := range []int{s.h, s.w} {
		k, st, p, d := int64(kernel[axis]), int64(stride[axis]), int64(padding[axis]), int64(dilation[axis])
		if k < 1 || k > 1<<31-1 || st < 1 || st > 1<<31-1 || d < 1 || d > 1<<31-1 || p < 0 || p > k/2 {
			return s, fmt.Errorf("autograd: invalid pooling kernel, stride, padding or dilation")
		}
		numerator := int64(input) + 2*p - d*(k-1) - 1
		if ceil {
			numerator += st - 1
		}
		dim := poolFloor(numerator, st) + 1
		if ceil && (dim-1)*st >= int64(input)+p {
			dim--
		}
		if dim < 1 || dim > 1<<31-1 {
			return s, fmt.Errorf("autograd: pooling output is empty or oversized")
		}
		output[axis] = int(dim)
	}
	s.oh, s.ow = output[0], output[1]
	return s, pool2dElements(s.outputShape())
}

func (s pool2dSpec) outputShape() []int { return []int{s.batch, s.channels, s.oh, s.ow} }

// Bounds are clipped before conversion back to int so dilated windows and
// adaptive products remain safe even near the common signed 32-bit limit.
func (s pool2dSpec) window(y, x, kind int) (hs, he, ws, we, dh, dw int, divisor float32) {
	var start, end [2]int64
	steps := [2]int64{1, 1}
	area := int64(1)
	for axis, position := range []int{y, x} {
		input, output := s.h, s.oh
		if axis == 1 {
			input, output = s.w, s.ow
		}
		if kind == 2 {
			start[axis] = int64(position) * int64(input) / int64(output)
			end[axis] = (int64(position+1)*int64(input) + int64(output) - 1) / int64(output)
		} else {
			st, p, k, d := int64(s.stride[axis]), int64(s.padding[axis]), int64(s.kernel[axis]), int64(s.dilation[axis])
			start[axis] = int64(position)*st - p
			end[axis] = start[axis] + (k-1)*d + 1
			if kind == 0 {
				steps[axis] = d
				if start[axis] < 0 {
					start[axis] += ((-start[axis] + d - 1) / d) * d
				}
			} else {
				end[axis] = min(end[axis], int64(input)+p)
				if !s.excludePad {
					area *= end[axis] - start[axis]
				}
				start[axis] = max(start[axis], 0)
			}
			end[axis] = min(end[axis], int64(input))
			if kind == 0 {
				start[axis] = min(start[axis], int64(input))
				end[axis] = max(end[axis], 0)
			}
		}
		if kind == 2 || s.excludePad {
			area *= end[axis] - start[axis]
		}
	}
	if s.divisor != 0 {
		area = int64(s.divisor)
	}
	return int(start[0]), int(end[0]), int(start[1]), int(end[1]), int(steps[0]), int(steps[1]), float32(area)
}

func cpuPool2d(x *Tensor, s pool2dSpec, kind int) *Tensor {
	n := numel(s.outputShape())
	out := cpuAlloc(n)
	var indices []int
	_, recording := graphRecording([]*Tensor{x})
	if kind == 0 && recording {
		indices = make([]int, n)
	}
	parallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			plane, position := i/(s.oh*s.ow), i%(s.oh*s.ow)
			hs, he, ws, we, dh, dw, divisor := s.window(position/s.ow, position%s.ow, kind)
			value := float32(0)
			index := -1
			if kind == 0 {
				value = float32(math.Inf(-1))
			}
			for h := int64(hs); h < int64(he); h += int64(dh) {
				for w := int64(ws); w < int64(we); w += int64(dw) {
					p := (plane*s.h+int(h))*s.w + int(w)
					v := x.Data[p]
					if kind == 0 {
						if index < 0 || v > value || math.IsNaN(float64(v)) {
							value, index = v, p
						}
					} else {
						value += v
					}
				}
			}
			if kind != 0 {
				value /= divisor
			}
			out[i] = value
			if indices != nil {
				indices[i] = index
			}
		}
	})
	return result(out, s.outputShape(), []*Tensor{x}, func(g []float32) {
		if !x.RequiresGrad {
			return
		}
		dx := cpuAlloc(x.Numel())
		defer cpuRelease(dx)
		// Shard by N*C, rather than windows: overlapping windows in a plane
		// accumulate serially while different planes have disjoint gradients.
		spatial := s.oh * s.ow
		parallelFor(n, func(start, end int) {
			// Use output work to choose the worker count, but round partitions
			// to whole planes so no input gradient is written by two workers.
			first, last := start/spatial, end/spatial
			if start%spatial != 0 {
				first++
			}
			if end%spatial != 0 {
				last++
			}
			for plane := first; plane < last; plane++ {
				for position := 0; position < s.oh*s.ow; position++ {
					i := plane*s.oh*s.ow + position
					if kind == 0 {
						if indices[i] >= 0 {
							dx[indices[i]] += g[i]
						}
						continue
					}
					hs, he, ws, we, dh, dw, divisor := s.window(position/s.ow, position%s.ow, kind)
					v := g[i] / divisor
					for h := hs; h < he; h += dh {
						for w := ws; w < we; w += dw {
							dx[(plane*s.h+h)*s.w+w] += v
						}
					}
				}
			}
		})
		x.addGrad(dx)
	})
}
