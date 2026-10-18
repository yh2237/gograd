package autograd

import (
	"fmt"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	_ "github.com/yh2237/gograd/vulkan" // Lazy registration; importing autograd does not load a driver.
)

func inferenceDevice(device tensor.Device) bool { return device != tensor.CPU && device != tensor.CUDA }

func newInferenceTensor(values []float32, shape []int, device tensor.Device, grad, zero bool) (*Tensor, error) {
	n, err := gpu.Elements(shape...)
	if err != nil {
		return nil, err
	}
	if !zero && len(values) != n {
		return nil, fmt.Errorf("%w: shape/data mismatch", gpu.ErrInvalidArgument)
	}
	b, err := gpu.Get(string(device))
	if err != nil {
		return nil, err
	}
	var buffer *gpu.Buffer
	if zero {
		buffer, err = b.Alloc(n)
	} else {
		buffer, err = b.Upload(values)
	}
	if err != nil {
		return nil, err
	}
	if err = gpu.CheckBuffer(b, buffer, n); err == nil && zero {
		err = b.Zero(buffer)
	}
	if err != nil {
		buffer.Close()
		return nil, err
	}
	return &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), Device: device, DType: Float32, RequiresGrad: grad, storage: inferenceStorage(buffer)}, nil
}

// Validate recording and all operands before materialization or allocation.
// Factory RequiresGrad flags are preserved; NoGrad on the appropriate context
// is what permits inference with ordinary trainable module parameters.
func inferenceOperands(inputs ...*Tensor) (gpu.Backend, *ExecutionContext) {
	same(inputs[0], inputs[1:]...)
	context, req := graphRecording(inputs)
	if req {
		panic(fmt.Errorf("%w: forward on %s requires NoGrad", gpu.ErrInferenceOnly, inputs[0].Device))
	}
	var backend gpu.Backend
	for _, t := range inputs {
		if t.DType != Float32 {
			panic(fmt.Errorf("%w: GPU inference requires float32", gpu.ErrUnsupported))
		}
		if t.storage == nil || t.storage.gpuBuffer == nil {
			panic(fmt.Errorf("%w: missing GPU storage", gpu.ErrBackendMismatch))
		}
		owner := t.storage.gpuBuffer.Backend()
		if backend != nil && backend != owner {
			panic(gpu.ErrBackendMismatch)
		}
		backend = owner
		if _, err := inferenceView(t).Validate(owner); err != nil {
			panic(err)
		}
	}
	return backend, context
}

func inferenceResult(buffer *gpu.Buffer, shape []int, context *ExecutionContext, parents ...*Tensor) *Tensor {
	r := &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: parents[0].Device, storage: inferenceStorage(buffer), execution: context, intermediate: true}
	r.attachParents(parents)
	return r
}

func inferenceBinary(name string, a, b *Tensor) *Tensor {
	if name != "add" && name != "mul" {
		panic(fmt.Errorf("%w: %s on %s", gpu.ErrUnsupported, name, a.Device))
	}
	backend, context := inferenceOperands(a, b)
	shape, err := gpu.BroadcastShape(a.Shape, b.Shape)
	if err != nil {
		panic(err)
	}
	n, err := gpu.Elements(shape...)
	if err != nil {
		panic(err)
	}
	av, err := inferenceView(a).BroadcastTo(shape)
	if err != nil {
		panic(err)
	}
	bv, err := inferenceView(b).BroadcastTo(shape)
	if err != nil {
		panic(err)
	}
	x, releaseA := inferenceBuffer(backend, av)
	defer releaseA()
	y, releaseB := inferenceBuffer(backend, bv)
	defer releaseB()
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
	if name == "add" {
		err = backend.Add(out, x, y, n)
	} else {
		err = backend.Mul(out, x, y, n)
	}
	if err != nil {
		panic(err)
	}
	return inferenceResult(out, shape, context, a, b)
}

func inferenceMulScalar(a *Tensor, scalar float32) *Tensor {
	backend, context := inferenceOperands(a)
	x, release := inferenceBuffer(backend, inferenceView(a))
	defer release()
	out, err := backend.Alloc(a.Numel())
	if err != nil {
		panic(err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.Close()
			panic(r)
		}
	}()
	if err := backend.MulScalar(out, x, a.Numel(), scalar); err != nil {
		panic(err)
	}
	return inferenceResult(out, a.Shape, context, a)
}

func detachInference(t *Tensor) *Tensor {
	backend := t.storage.gpuBuffer.Backend()
	v := inferenceView(t)
	if _, err := v.Validate(backend); err != nil {
		panic(err)
	}
	out, err := backend.Alloc(t.Numel())
	if err != nil {
		panic(err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.Close()
			panic(r)
		}
	}()
	if t.IsContiguous() {
		copier, ok := backend.(gpu.Copier)
		if !ok {
			panic(fmt.Errorf("%w: device copy", gpu.ErrUnsupported))
		}
		if err := copier.Copy(out, t.storage.gpuBuffer, t.Numel()); err != nil {
			panic(err)
		}
	} else {
		views, ok := backend.(gpu.ViewBackend)
		if !ok {
			panic(fmt.Errorf("%w: device view copy", gpu.ErrUnsupported))
		}
		if err := views.Gather(out, v); err != nil {
			panic(err)
		}
	}
	return &Tensor{Shape: append([]int(nil), t.Shape...), Strides: strides(t.Shape), DType: t.DType, Device: t.Device, storage: inferenceStorage(out), execution: t.execution}
}

func inferenceView(t *Tensor) gpu.View {
	return gpu.View{Buffer: t.storage.gpuBuffer, Shape: t.Shape, Strides: t.Strides, Offset: t.Offset}
}

// Scratch buffers belong only to this dispatch, not to the autograd graph.
// Backend release semantics keep queued device reads safe without a download.
func inferenceBuffer(backend gpu.Backend, v gpu.View) (*gpu.Buffer, func()) {
	n, err := v.Validate(backend)
	if err != nil {
		panic(err)
	}
	if v.Offset == 0 && v.Buffer.Len() == n && v.Compact() {
		return v.Buffer, func() {}
	}
	views, ok := backend.(gpu.ViewBackend)
	if !ok {
		panic(fmt.Errorf("%w: GPU view materialization", gpu.ErrUnsupported))
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
	if err := views.Gather(out, v); err != nil {
		panic(err)
	}
	return out, out.Close
}

func contiguousInference(t *Tensor) *Tensor {
	backend, context := inferenceOperands(t)
	views, ok := backend.(gpu.ViewBackend)
	if !ok {
		panic(fmt.Errorf("%w: GPU view materialization", gpu.ErrUnsupported))
	}
	out, err := backend.Alloc(t.Numel())
	if err != nil {
		panic(err)
	}
	defer func() {
		if r := recover(); r != nil {
			out.Close()
			panic(r)
		}
	}()
	if err := views.Gather(out, inferenceView(t)); err != nil {
		panic(err)
	}
	return inferenceResult(out, t.Shape, context, t)
}

func copyInferenceView(t *Tensor, values []float32) error {
	backend := t.storage.gpuBuffer.Backend()
	v := inferenceView(t)
	if _, err := v.ValidateWrite(backend); err != nil {
		return err
	}
	views, ok := backend.(gpu.ViewBackend)
	if !ok {
		return fmt.Errorf("%w: GPU view write", gpu.ErrUnsupported)
	}
	src, err := backend.Upload(values)
	if err != nil {
		return err
	}
	defer src.Close()
	return views.Scatter(v, src)
}

func inferenceMatMul(a, b *Tensor) *Tensor {
	backend, context := inferenceOperands(a, b)
	shape, err := gpu.MatMulShape(a.Shape, b.Shape)
	if err != nil {
		panic(err)
	}
	n, err := gpu.Elements(shape...)
	if err != nil {
		panic(err)
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
	if matrices, ok := backend.(gpu.ViewMatMuler); ok {
		err = matrices.MatMulViews(out, inferenceView(a), inferenceView(b))
	} else {
		if len(a.Shape) != 2 || len(b.Shape) != 2 {
			panic(fmt.Errorf("%w: batched GPU GEMM", gpu.ErrUnsupported))
		}
		x, releaseA := inferenceBuffer(backend, inferenceView(a))
		defer releaseA()
		y, releaseB := inferenceBuffer(backend, inferenceView(b))
		defer releaseB()
		err = backend.MatMul(out, x, y, a.Shape[0], b.Shape[1], a.Shape[1], false, false)
	}
	if err != nil {
		panic(err)
	}
	return inferenceResult(out, shape, context, a, b)
}
