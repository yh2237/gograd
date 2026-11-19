package gpu

import "fmt"

type BinaryOp uint8

const (
	BinarySub BinaryOp = iota
	BinaryDiv
	BinaryMax
	BinaryMin
)

type BinaryOpBackend interface {
	BinaryOp(op BinaryOp, out, a, b *Buffer, n int) error
}

type ViewBinaryOpBackend interface {
	BinaryOpViews(op BinaryOp, out *Buffer, a, b View) error
}

type ScalarOp uint8

const (
	ScalarAdd ScalarOp = iota
	ScalarSub
	ScalarDiv
	ScalarPow
)

type ScalarOpBackend interface {
	ScalarOp(op ScalarOp, out, a *Buffer, n int, value float32) error
}

type UnaryOp uint8

const (
	UnaryExp UnaryOp = iota
	UnaryLog
	UnaryAbs
	UnaryTanh
	UnaryReLU
	UnarySin
	UnaryCos
	UnarySqrt
	UnarySigmoid
	UnarySoftplus
	UnaryErf
)

type UnaryOpBackend interface {
	UnaryOp(op UnaryOp, out, a *Buffer, n int) error
	Clamp(out, a *Buffer, n int, lo, hi float32) error
}

type LogSoftmaxBackend interface {
	LogSoftmax(out, a *Buffer, rows, cols int) error
}

type ReduceOp uint8

const (
	ReduceSum ReduceOp = iota
	ReduceMean
)

type ReduceBackend interface {
	Reduce(op ReduceOp, out, a *Buffer, shape, axes []int) error
}

type Conv2dOptions struct {
	Batch, InChannels, Height, Width int
	OutChannels, KernelH, KernelW    int
	StrideH, StrideW, PadH, PadW     int
	DilationH, DilationW, Groups     int
}

func (o Conv2dOptions) OutputShape() (int, int, error) {
	for _, d := range []int{o.Batch, o.InChannels, o.Height, o.Width, o.OutChannels, o.KernelH, o.KernelW, o.Groups} {
		if d < 1 || d > maxElements {
			return 0, 0, fmt.Errorf("%w: Conv2d dimensions", ErrInvalidArgument)
		}
	}
	if o.InChannels%o.Groups != 0 || o.OutChannels%o.Groups != 0 {
		return 0, 0, fmt.Errorf("%w: Conv2d groups", ErrInvalidArgument)
	}
	for _, d := range []int{o.StrideH, o.StrideW, o.DilationH, o.DilationW} {
		if d < 1 || d > maxElements {
			return 0, 0, fmt.Errorf("%w: Conv2d stride/dilation", ErrInvalidArgument)
		}
	}
	for _, d := range []int{o.PadH, o.PadW} {
		if d < 0 || d > maxElements {
			return 0, 0, fmt.Errorf("%w: Conv2d padding", ErrInvalidArgument)
		}
	}
	axis := func(in, k, s, p, dil int) (int, error) {
		span := int64(dil)*int64(k-1) + 1
		padded := int64(in) + 2*int64(p)
		if span > int64(maxElements) || padded > int64(maxElements) || span > padded {
			return 0, fmt.Errorf("%w: Conv2d window exceeds input", ErrInvalidArgument)
		}
		out := (padded-span)/int64(s) + 1
		if out < 1 || out > int64(maxElements) {
			return 0, fmt.Errorf("%w: Conv2d output size", ErrInvalidArgument)
		}
		return int(out), nil
	}
	oh, err := axis(o.Height, o.KernelH, o.StrideH, o.PadH, o.DilationH)
	if err != nil {
		return 0, 0, err
	}
	ow, err := axis(o.Width, o.KernelW, o.StrideW, o.PadW, o.DilationW)
	if err != nil {
		return 0, 0, err
	}
	return oh, ow, nil
}

type Pool2dOptions struct {
	Batch, Channels, Height, Width     int
	KernelH, KernelW, StrideH, StrideW int
	PadH, PadW, EndPadH, EndPadW       int
	DilationH, DilationW               int
	OutH, OutW                         int
	IncludePad                         bool
}

type VisionBackend interface {
	Conv2d(out, x, weight, bias *Buffer, options Conv2dOptions) error
	MaxPool2d(out, x *Buffer, options Pool2dOptions) error
	AvgPool2d(out, x *Buffer, options Pool2dOptions) error
}

type BatchNormBackend interface {
	BatchNormEval(out, x, weight, bias, mean, variance *Buffer, outer, channels, inner int, eps float32) error
}
