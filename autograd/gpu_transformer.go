package autograd

import (
	"errors"
	"fmt"
	"math"
	"slices"

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
	return inferenceSoftmaxAxis(a, axis, false)
}

func inferenceLogSoftmax(a *Tensor, axis int) *Tensor {
	return inferenceSoftmaxAxis(a, axis, true)
}

func inferenceSoftmaxAxis(a *Tensor, axis int, logmode bool) *Tensor {
	backend, _ := inferenceOperands(a)
	var ls gpu.LogSoftmaxBackend
	if logmode {
		var ok bool
		ls, ok = backend.(gpu.LogSoftmaxBackend)
		if !ok {
			panic(fmt.Errorf("%w: log_softmax on %s", gpu.ErrUnsupported, a.Device))
		}
	}
	if axis < 0 {
		axis += len(a.Shape)
	}
	if axis < 0 || axis >= len(a.Shape) || a.Shape[axis] == 0 {
		panic(fmt.Errorf("%w: softmax axis", gpu.ErrInvalidArgument))
	}
	v := inferenceView(a)
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
		if logmode {
			return ls.LogSoftmax(out, inputs[0], a.Numel()/cols, cols)
		}
		return b.Softmax(out, inputs[0], a.Numel()/cols, cols)
	})
	if axis == len(a.Shape)-1 {
		return out
	}
	res := Permute(out, inverse...)
	out.releaseRequested.Store(true)
	return res
}

func inferenceReduce(op gpu.ReduceOp, name string, a *Tensor, axes ...int) *Tensor {
	backend, _ := inferenceOperands(a)
	red, ok := backend.(gpu.ReduceBackend)
	if !ok {
		panic(fmt.Errorf("%w: %s on %s", gpu.ErrUnsupported, name, a.Device))
	}
	selected := make([]bool, len(a.Shape))
	normAxes := make([]int, 0, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += len(selected)
		}
		if ax < 0 || ax >= len(selected) || selected[ax] {
			panic(fmt.Errorf("%w: invalid axis", gpu.ErrInvalidArgument))
		}
		selected[ax] = true
		normAxes = append(normAxes, ax)
	}
	outShape := make([]int, 0, len(a.Shape)-len(normAxes))
	for i, d := range a.Shape {
		if !selected[i] {
			outShape = append(outShape, d)
		}
	}
	return inferenceForward(outShape, []*Tensor{a}, nil, func(_ gpu.Backend, out *gpu.Buffer, inputs []*gpu.Buffer) error {
		return red.Reduce(op, out, inputs[0], a.Shape, normAxes)
	})
}

func inferenceConv2d(x, w, b *Tensor, options Conv2dOptions) *Tensor {
	inputs := []*Tensor{x, w}
	if b != nil {
		inputs = append(inputs, b)
	}
	backend, _ := inferenceOperands(inputs...)
	vis, ok := backend.(gpu.VisionBackend)
	if !ok {
		panic(fmt.Errorf("%w: conv2d on %s", gpu.ErrUnsupported, x.Device))
	}
	spec, err := conv2dShape(x.Shape, w.Shape, options)
	if err != nil {
		panic(err)
	}
	if b != nil && (len(b.Shape) != 1 || b.Shape[0] != spec.co) {
		panic("autograd: Conv2d bias must have shape [out_channels]")
	}
	o := gpu.Conv2dOptions{
		Batch: spec.batch, InChannels: spec.ci, Height: spec.h, Width: spec.width,
		OutChannels: spec.co, KernelH: spec.kh, KernelW: spec.kw,
		StrideH: spec.options.Stride[0], StrideW: spec.options.Stride[1],
		PadH: spec.options.Padding[0], PadW: spec.options.Padding[1],
		DilationH: spec.options.Dilation[0], DilationW: spec.options.Dilation[1],
		Groups: spec.options.Groups,
	}
	return inferenceForward(spec.outputShape(), inputs, nil, func(_ gpu.Backend, out *gpu.Buffer, values []*gpu.Buffer) error {
		var biasBuf *gpu.Buffer
		if b != nil {
			biasBuf = values[2]
		}
		return vis.Conv2d(out, values[0], values[1], biasBuf, o)
	})
}

func poolEndPad(in, out, k, st, p, d int) int {
	need := (out-1)*st + d*(k-1) + 1 - in - p
	return max(p, need)
}

func inferenceMaxPool2d(x *Tensor, options MaxPool2dOptions) *Tensor {
	backend, _ := inferenceOperands(x)
	vis, ok := backend.(gpu.VisionBackend)
	if !ok {
		panic(fmt.Errorf("%w: max_pool2d on %s", gpu.ErrUnsupported, x.Device))
	}
	s, err := pool2dShape(x.Shape, options.KernelSize, options.Stride, options.Padding, options.Dilation, options.CeilMode)
	if err != nil {
		panic(err)
	}
	po := gpu.Pool2dOptions{
		Batch: s.batch, Channels: s.channels, Height: s.h, Width: s.w,
		KernelH: s.kernel[0], KernelW: s.kernel[1],
		StrideH: s.stride[0], StrideW: s.stride[1],
		PadH: s.padding[0], PadW: s.padding[1],
		EndPadH:   poolEndPad(s.h, s.oh, s.kernel[0], s.stride[0], s.padding[0], s.dilation[0]),
		EndPadW:   poolEndPad(s.w, s.ow, s.kernel[1], s.stride[1], s.padding[1], s.dilation[1]),
		DilationH: s.dilation[0], DilationW: s.dilation[1],
		OutH: s.oh, OutW: s.ow,
	}
	return inferenceForward(s.outputShape(), []*Tensor{x}, nil, func(_ gpu.Backend, out *gpu.Buffer, values []*gpu.Buffer) error {
		return vis.MaxPool2d(out, values[0], po)
	})
}

func inferenceAvgPool2d(x *Tensor, options AvgPool2dOptions) *Tensor {
	backend, _ := inferenceOperands(x)
	vis, ok := backend.(gpu.VisionBackend)
	if !ok {
		panic(fmt.Errorf("%w: avg_pool2d on %s", gpu.ErrUnsupported, x.Device))
	}
	s, err := pool2dShape(x.Shape, options.KernelSize, options.Stride, options.Padding, [2]int{1, 1}, options.CeilMode)
	if err != nil {
		panic(err)
	}
	if options.DivisorOverride < 0 || options.DivisorOverride > 1<<31-1 {
		panic("autograd: invalid AvgPool2d divisor override")
	}
	if options.DivisorOverride != 0 {
		panic(fmt.Errorf("%w: AvgPool2d DivisorOverride on %s", gpu.ErrUnsupported, x.Device))
	}
	endPadH := poolEndPad(s.h, s.oh, s.kernel[0], s.stride[0], s.padding[0], 1)
	endPadW := poolEndPad(s.w, s.ow, s.kernel[1], s.stride[1], s.padding[1], 1)
	if (endPadH > s.padding[0] || endPadW > s.padding[1]) && !options.ExcludePad {
		panic(fmt.Errorf("%w: AvgPool2d CeilMode with included padding on %s", gpu.ErrUnsupported, x.Device))
	}
	po := gpu.Pool2dOptions{
		Batch: s.batch, Channels: s.channels, Height: s.h, Width: s.w,
		KernelH: s.kernel[0], KernelW: s.kernel[1],
		StrideH: s.stride[0], StrideW: s.stride[1],
		PadH: s.padding[0], PadW: s.padding[1],
		EndPadH: endPadH, EndPadW: endPadW,
		DilationH: 1, DilationW: 1,
		OutH: s.oh, OutW: s.ow,
		IncludePad: !options.ExcludePad,
	}
	return inferenceForward(s.outputShape(), []*Tensor{x}, nil, func(_ gpu.Backend, out *gpu.Buffer, values []*gpu.Buffer) error {
		return vis.AvgPool2d(out, values[0], po)
	})
}

func inferenceAdaptiveAvgPool2d(x *Tensor, outputSize [2]int) *Tensor {
	backend, _ := inferenceOperands(x)
	if _, ok := backend.(gpu.VisionBackend); !ok {
		panic(fmt.Errorf("%w: adaptive_avg_pool2d on %s", gpu.ErrUnsupported, x.Device))
	}
	s, err := adaptivePool2dShape(x.Shape, outputSize)
	if err != nil {
		panic(err)
	}
	if s.h%s.oh != 0 || s.w%s.ow != 0 {
		panic(fmt.Errorf("%w: non-divisible AdaptiveAvgPool2d on %s", gpu.ErrUnsupported, x.Device))
	}
	kh, kw := s.h/s.oh, s.w/s.ow
	return inferenceAvgPool2d(x, AvgPool2dOptions{KernelSize: [2]int{kh, kw}, Stride: [2]int{kh, kw}})
}

func inferenceBatchNormEval(x, weight, bias, mean, variance, count *Tensor, options BatchNormOptions) *Tensor {
	if options.Training {
		panic(gpu.ErrInferenceOnly)
	}
	all := []*Tensor{x, weight, bias, mean, variance, count}
	live := make([]*Tensor, 0, len(all))
	for _, t := range all {
		if t != nil {
			live = append(live, t)
		}
	}
	backend, context := inferenceOperands(live...)
	bn, ok := backend.(gpu.BatchNormBackend)
	if !ok {
		panic(fmt.Errorf("%w: batch_norm on %s", gpu.ErrUnsupported, x.Device))
	}
	o, err := options.normalized()
	if err != nil {
		panic(err)
	}
	s, err := batchNormShape(x.Shape, o.ChannelsLast, false)
	if err != nil {
		panic(err)
	}
	if mean == nil || variance == nil {
		panic("autograd: BatchNorm requires paired running statistics for evaluation")
	}
	for _, v := range []*Tensor{weight, bias, mean, variance} {
		if v != nil && (len(v.Shape) != 1 || v.Shape[0] != s.channels) {
			panic("autograd: BatchNorm channel vector shape")
		}
	}
	for _, v := range []*Tensor{mean, variance, count} {
		if v != nil && (v.RequiresGrad || !v.IsContiguous()) {
			panic("autograd: BatchNorm state must be contiguous non-gradient buffers")
		}
	}
	if count != nil && len(count.Shape) != 0 {
		panic("autograd: BatchNorm counter requires scalar tracked state")
	}
	for i := 3; i < len(all); i++ {
		if all[i] == nil {
			continue
		}
		for j, v := range all {
			if j != i && v != nil && all[i].storage == v.storage {
				panic("autograd: BatchNorm running state aliases an input")
			}
		}
	}
	inputs := []*Tensor{x, mean, variance}
	weightIdx, biasIdx := -1, -1
	if weight != nil {
		weightIdx = len(inputs)
		inputs = append(inputs, weight)
	}
	if bias != nil {
		biasIdx = len(inputs)
		inputs = append(inputs, bias)
	}
	out := inferenceForward(x.Shape, inputs, nil, func(_ gpu.Backend, out *gpu.Buffer, values []*gpu.Buffer) error {
		var wBuf, bBuf *gpu.Buffer
		if weightIdx >= 0 {
			wBuf = values[weightIdx]
		}
		if biasIdx >= 0 {
			bBuf = values[biasIdx]
		}
		return bn.BatchNormEval(out, values[0], wBuf, bBuf, values[1], values[2], s.outer, s.channels, s.inner, o.Eps)
	})
	if out.execution == nil {
		out.execution = context
	}
	return out
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

func inferenceEmbedding(weight *Tensor, ids []int, shape []int, paddingIdx int) *Tensor {
	backend, context := inferenceOperands(weight)
	if len(weight.Shape) != 2 || numel(shape) != len(ids) {
		panic("autograd: embedding shape")
	}
	if paddingIdx < -1 || paddingIdx >= weight.Shape[0] {
		panic("autograd: padding index")
	}
	for _, id := range ids {
		if id < 0 || id >= weight.Shape[0] {
			panic("autograd: embedding index")
		}
	}
	dim := weight.Shape[1]
	outShape := append(append([]int(nil), shape...), dim)
	n, err := gpu.Elements(outShape...)
	if err != nil {
		panic(err)
	}
	views, ok := backend.(gpu.ViewBackend)
	if !ok {
		panic(fmt.Errorf("%w: Embedding needs device view copies", gpu.ErrUnsupported))
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
	if n == 0 {
		return inferenceResult(out, outShape, context, weight)
	}
	src, release := inferenceBuffer(backend, inferenceView(weight))
	defer release()
	contiguous := true
	for i := 1; i < len(ids); i++ {
		if ids[i] != ids[i-1]+1 {
			contiguous = false
			break
		}
	}
	if contiguous {
		if err := views.Gather(out, gpu.View{Buffer: src, Shape: []int{n}, Strides: []int{1}, Offset: ids[0] * dim}); err != nil {
			panic(err)
		}
		return inferenceResult(out, outShape, context, weight)
	}
	tmp, err := backend.Alloc(dim)
	if err != nil {
		panic(err)
	}
	defer tmp.Close()
	for i, id := range ids {
		if err := views.Gather(tmp, gpu.View{Buffer: src, Shape: []int{dim}, Strides: []int{1}, Offset: id * dim}); err != nil {
			panic(err)
		}
		if err := views.Scatter(gpu.View{Buffer: out, Shape: []int{dim}, Strides: []int{1}, Offset: i * dim}, tmp); err != nil {
			panic(err)
		}
	}
	return inferenceResult(out, outShape, context, weight)
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
	if out, ok := inferenceFusedBias(x, bias); ok {
		return out
	}
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
	if out, ok := inferenceFusedBias(x, bias, residual); ok {
		return out
	}
	sum := Add(x, bias)
	defer func() {
		if r := recover(); r != nil {
			sum.Close()
			panic(r)
		}
	}()
	return Add(sum, residual)
}

// Fusion is optional and deliberately narrower than general broadcasting. Keep
// the existing Add/GELU composition for other shapes or ErrUnsupported. Scratch
// for non-contiguous inputs stays local; only the output enters graph storage.
func inferenceFusedBias(inputs ...*Tensor) (*Tensor, bool) {
	backend, context := inferenceOperands(inputs...)
	x, bias := inputs[0], inputs[1]
	fused, ok := backend.(gpu.FusedBiasBackend)
	if !ok || len(x.Shape) == 0 || len(bias.Shape) != 1 {
		return nil, false
	}
	cols := x.Shape[len(x.Shape)-1]
	if cols < 1 || bias.Shape[0] != cols || (len(inputs) == 3 && !slices.Equal(x.Shape, inputs[2].Shape)) {
		return nil, false
	}
	values := make([]*gpu.Buffer, len(inputs))
	for i, v := range inputs {
		value, release := inferenceBuffer(backend, inferenceView(v))
		defer release()
		values[i] = value
	}
	out, err := backend.Alloc(x.Numel())
	if err != nil {
		panic(err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.Close()
			panic(r)
		}
	}()
	rows := x.Numel() / cols
	if len(inputs) == 2 {
		err = fused.BiasGELU(out, values[0], values[1], rows, cols)
	} else {
		err = fused.BiasResidual(out, values[0], values[1], values[2], rows, cols)
	}
	if errors.Is(err, gpu.ErrUnsupported) {
		out.Close()
		return nil, false
	}
	if err != nil {
		panic(err)
	}
	return inferenceResult(out, x.Shape, context, inputs...), true
}
