package autograd

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/gpu"
)

// inferenceForward keeps materialization scratch local to this dispatch. Only
// the final output becomes graph storage; all temporary buffers close on error.
func inferenceForward(shape []int, inputs []*Tensor, views []gpu.View, run func(gpu.Backend, *gpu.Buffer, []*gpu.Buffer) error) *Tensor {
	backend, context := inferenceOperands(inputs...)
	n, err := gpu.Elements(shape...)
	if err != nil {
		panic(err)
	}
	buffers := make([]*gpu.Buffer, len(inputs))
	for i, t := range inputs {
		v := inferenceView(t)
		if views != nil {
			v = views[i]
		}
		buffer, release := inferenceBuffer(backend, v)
		defer release()
		buffers[i] = buffer
	}
	out, err := backend.Alloc(n)
	if err != nil {
		panic(err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.Close()
			panic(r)
		}
	}()
	if err := run(backend, out, buffers); err != nil {
		panic(err)
	}
	return inferenceResult(out, shape, context, inputs...)
}

func inferenceGELU(a *Tensor, approx bool) *Tensor {
	return inferenceForward(a.Shape, []*Tensor{a}, nil, func(b gpu.Backend, out *gpu.Buffer, inputs []*gpu.Buffer) error {
		if approx {
			return b.GELU(out, inputs[0], a.Numel())
		}
		exact, ok := b.(gpu.ExactGELUBackend)
		if !ok {
			return fmt.Errorf("%w: erf-based GELU", gpu.ErrUnsupported)
		}
		return exact.GELUExact(out, inputs[0], a.Numel())
	})
}

func inferenceSoftmax(a *Tensor, axis int) *Tensor {
	a.assertOpen()
	if axis < 0 {
		axis += len(a.Shape)
	}
	if axis < 0 || axis >= len(a.Shape) || a.Shape[axis] == 0 {
		panic(fmt.Errorf("%w: softmax axis", gpu.ErrInvalidArgument))
	}
	v := inferenceView(a)
	// The backend ABI reduces contiguous rows. Move a non-last axis to the
	// end with metadata, gather once, then return an inverse-permuted view.
	axes := make([]int, 0, len(a.Shape))
	for i := range a.Shape {
		if i != axis {
			axes = append(axes, i)
		}
	}
	axes = append(axes, axis)
	shape, stride, inverse := make([]int, len(axes)), make([]int, len(axes)), make([]int, len(axes))
	for i, ax := range axes {
		shape[i], stride[i], inverse[ax] = a.Shape[ax], a.Strides[ax], i
	}
	v.Shape, v.Strides = shape, stride
	cols := a.Shape[axis]
	out := inferenceForward(shape, []*Tensor{a}, []gpu.View{v}, func(b gpu.Backend, out *gpu.Buffer, inputs []*gpu.Buffer) error {
		return b.Softmax(out, inputs[0], a.Numel()/cols, cols)
	})
	if axis == len(a.Shape)-1 {
		return out
	}
	return Permute(out, inverse...)
}

func inferenceLayerNorm(x, w, bias *Tensor, eps float32) *Tensor {
	if len(x.Shape) < 1 {
		panic(fmt.Errorf("%w: LayerNorm rank", gpu.ErrInvalidArgument))
	}
	cols := x.Shape[len(x.Shape)-1]
	if cols < 1 || len(w.Shape) != 1 || len(bias.Shape) != 1 || w.Numel() != cols || bias.Numel() != cols || !validInferenceEps(eps) {
		panic(fmt.Errorf("%w: LayerNorm shape/epsilon", gpu.ErrInvalidArgument))
	}
	return inferenceForward(x.Shape, []*Tensor{x, w, bias}, nil, func(b gpu.Backend, out *gpu.Buffer, inputs []*gpu.Buffer) error {
		return b.LayerNorm(out, inputs[0], inputs[1], inputs[2], x.Numel()/cols, cols, eps)
	})
}

func inferenceGroupNorm(x, w, bias *Tensor, groups int, eps float32) *Tensor {
	if len(x.Shape) != 3 || x.Shape[1] < 1 || x.Shape[2] < 1 || groups < 1 || x.Shape[2]%groups != 0 || len(w.Shape) != 1 || len(bias.Shape) != 1 || w.Numel() != x.Shape[2] || bias.Numel() != x.Shape[2] || !validInferenceEps(eps) {
		panic(fmt.Errorf("%w: GroupNorm shape/epsilon", gpu.ErrInvalidArgument))
	}
	return inferenceForward(x.Shape, []*Tensor{x, w, bias}, nil, func(b gpu.Backend, out *gpu.Buffer, inputs []*gpu.Buffer) error {
		return b.GroupNorm(out, inputs[0], inputs[1], inputs[2], x.Shape[0], x.Shape[1], x.Shape[2], groups, eps)
	})
}

func inferenceConv1d(x, w, bias *Tensor, stride, padding, dilation int, samePadding bool) *Tensor {
	if len(x.Shape) != 3 || len(w.Shape) != 3 || x.Shape[2] != w.Shape[1] || (bias != nil && (len(bias.Shape) != 1 || bias.Numel() != w.Shape[0])) {
		panic(fmt.Errorf("%w: Conv1d shape", gpu.ErrInvalidArgument))
	}
	if samePadding {
		if w.Shape[2]%2 != 1 {
			panic(fmt.Errorf("%w: same-padded Conv1d needs odd kernel", gpu.ErrInvalidArgument))
		}
		if _, err := gpu.Elements(dilation); err != nil {
			panic(err)
		}
		pad := int64(w.Shape[2]/2) * int64(dilation)
		if pad > math.MaxInt32 {
			panic(gpu.ErrInvalidArgument)
		}
		padding = int(pad)
	}
	o := gpu.Conv1dOptions{Batch: x.Shape[0], InputLength: x.Shape[1], InChannels: x.Shape[2], OutChannels: w.Shape[0], Kernel: w.Shape[2], Stride: stride, Padding: padding, Dilation: dilation}
	length, err := o.OutputLength()
	if err != nil {
		panic(err)
	}
	inputs := []*Tensor{x, w}
	if bias != nil {
		inputs = append(inputs, bias)
	}
	return inferenceForward([]int{o.Batch, length, o.OutChannels}, inputs, nil, func(b gpu.Backend, out *gpu.Buffer, values []*gpu.Buffer) error {
		var biasBuffer *gpu.Buffer
		if bias != nil {
			biasBuffer = values[2]
		}
		return b.Conv1d(out, values[0], values[1], biasBuffer, o)
	})
}

func inferenceConcat(axis int, inputs ...*Tensor) *Tensor {
	backend, context := inferenceOperands(inputs...)
	shape := append([]int(nil), inputs[0].Shape...)
	if axis < 0 {
		axis += len(shape)
	}
	if axis < 0 || axis >= len(shape) {
		panic(fmt.Errorf("%w: Concat axis", gpu.ErrInvalidArgument))
	}
	shape[axis] = 0
	for _, x := range inputs {
		if len(x.Shape) != len(shape) {
			panic(fmt.Errorf("%w: Concat rank", gpu.ErrInvalidArgument))
		}
		for i, d := range x.Shape {
			if i != axis && d != shape[i] {
				panic(fmt.Errorf("%w: Concat shape", gpu.ErrInvalidArgument))
			}
		}
		shape[axis] += x.Shape[axis]
		if _, err := gpu.Elements(shape[axis]); err != nil {
			panic(err)
		}
	}
	n, err := gpu.Elements(shape...)
	if err != nil {
		panic(err)
	}
	views, ok := backend.(gpu.ViewBackend)
	if !ok {
		panic(fmt.Errorf("%w: Concat needs device view copies", gpu.ErrUnsupported))
	}
	out, err := backend.Alloc(n)
	if err != nil {
		panic(err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.Close()
			panic(r)
		}
	}()
	offset, st := 0, strides(shape)
	for _, x := range inputs {
		if x.Numel() != 0 {
			func() {
				src, release := inferenceBuffer(backend, inferenceView(x))
				defer release()
				if err := views.Scatter(gpu.View{Buffer: out, Shape: x.Shape, Strides: st, Offset: offset}, src); err != nil {
					panic(err)
				}
			}()
		}
		offset += x.Shape[axis] * st[axis]
	}
	return inferenceResult(out, shape, context, inputs...)
}

// The portable materialized attention path composes GEMM, a device mask add,
// and Softmax. FlashAttention remains unsupported on inference backends.
func inferenceAttention(q, k, v, mask *Tensor) *Tensor {
	inputs := []*Tensor{q, k, v}
	if mask != nil {
		inputs = append(inputs, mask)
	}
	inferenceOperands(inputs...)
	options := executionOptions(inputs...)
	if options.AttentionAlgorithm == "flash" {
		panic(fmt.Errorf("%w: FlashAttention", gpu.ErrUnsupported))
	}
	if len(q.Shape) < 2 || len(k.Shape) < 2 || len(v.Shape) < 2 || q.Shape[len(q.Shape)-1] < 1 || k.Shape[len(k.Shape)-2] < 1 || q.Shape[len(q.Shape)-1] != k.Shape[len(k.Shape)-1] || k.Shape[len(k.Shape)-2] != v.Shape[len(v.Shape)-2] {
		panic(fmt.Errorf("%w: attention dimensions", gpu.ErrInvalidArgument))
	}
	ks := append([]int(nil), k.Shape...)
	ks[len(ks)-2], ks[len(ks)-1] = ks[len(ks)-1], ks[len(ks)-2]
	scoresShape, err := gpu.MatMulShape(q.Shape, ks)
	if err != nil {
		panic(err)
	}
	if mask != nil {
		scoresShape, err = gpu.BroadcastShape(scoresShape, mask.Shape)
		if err != nil {
			panic(err)
		}
	}
	if _, err := gpu.MatMulShape(scoresShape, v.Shape); err != nil {
		panic(err)
	}
	// Close only nodes owned by this composition on failure, not the caller's
	// input graphs. Parent reference counts keep shared branches intact.
	var owned []*Tensor
	keep := func(t *Tensor) *Tensor { owned = append(owned, t); return t }
	defer func() {
		if r := recover(); r != nil {
			for i := len(owned) - 1; i >= 0; i-- {
				owned[i].Close()
			}
			panic(r)
		}
	}()
	kt := keep(Transpose(k, len(k.Shape)-2, len(k.Shape)-1))
	scores := keep(MatMul(q, kt))
	scores = keep(MulScalar(scores, 1/float32(math.Sqrt(float64(q.Shape[len(q.Shape)-1])))))
	if mask != nil {
		scores = keep(Add(scores, mask))
	}
	weights := keep(Softmax(scores, -1))
	return MatMul(weights, v)
}

func inferenceBiasGELU(x, bias *Tensor) *Tensor {
	inferenceOperands(x, bias)
	sum := Add(x, bias)
	defer func() {
		if r := recover(); r != nil {
			sum.Close()
			panic(r)
		}
	}()
	return GELU(sum, false)
}

func inferenceBiasResidual(x, bias, residual *Tensor) *Tensor {
	inferenceOperands(x, bias, residual)
	sum := Add(x, bias)
	defer func() {
		if r := recover(); r != nil {
			sum.Close()
			panic(r)
		}
	}()
	return Add(sum, residual)
}
