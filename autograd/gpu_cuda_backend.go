package autograd

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/kernels"
)

// Register only the inference adapter here. Existing CUDA training, BF16 and
// capture paths retain their execution options and gradient/lifetime handling.
// No driver is loaded by registration. Calls need the same pinned OS-thread
// context as the existing CUDA API (NewCUDAContext/NewCUDAContextOn).
func init() {
	if err := gpu.Register(&cudaInferenceBackend{}); err != nil {
		panic(err)
	}
}

// Non-zero size preserves distinct pointer identity for backend instances.
type cudaInferenceBackend struct{ identity byte }

type cudaInferenceMemory struct {
	buffer *cuda.Buffer
	stream *cuda.Stream
}

var _ gpu.Backend = (*cudaInferenceBackend)(nil)

func (*cudaInferenceBackend) Name() string    { return "cuda" }
func (*cudaInferenceBackend) Available() bool { return cuda.Available() }

// Existing CUDA helpers report driver/kernel failures by panicking with an
// error. Convert those errors at the new interface boundary, not programming
// panics such as a nil-pointer dereference.
func inferenceCUDA(fn func() error) (err error) {
	if !cuda.Available() {
		return gpu.ErrUnavailable
	}
	defer func() {
		if value := recover(); value != nil {
			if _, ok := value.(runtime.Error); ok {
				panic(value)
			}
			if failure, ok := value.(error); ok {
				err = fmt.Errorf("gpu cuda: %w", failure)
			} else {
				panic(value)
			}
		}
	}()
	return fn()
}

func (b *cudaInferenceBackend) memory(buffer *gpu.Buffer, n int) (*cudaInferenceMemory, error) {
	if err := gpu.CheckBuffer(b, buffer, n); err != nil {
		return nil, err
	}
	handle, err := buffer.Native(b)
	if err != nil {
		return nil, err
	}
	memory, ok := handle.(*cudaInferenceMemory)
	if !ok || memory == nil || memory.buffer == nil || memory.buffer.Size() != n*4 {
		return nil, fmt.Errorf("%w: invalid CUDA allocation", gpu.ErrInvalidArgument)
	}
	return memory, nil
}

// Transfers may be called on another thread/device. Restore both selections
// afterwards, and explicitly wait for the allocation's compute stream on reads.
func withInferenceMemory(memory *cudaInferenceMemory, fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return inferenceCUDA(func() error {
		previous, err := cuda.CurrentDevice()
		if err != nil {
			return err
		}
		if previous != memory.buffer.Device() {
			if err := cuda.SetDevice(memory.buffer.Device()); err != nil {
				return err
			}
			defer cuda.SetDevice(previous)
		}
		stream := cuda.CurrentStream()
		cuda.SetCurrentStream(memory.stream)
		defer cuda.SetCurrentStream(stream)
		return fn()
	})
}

func waitInferenceMemory(memory *cudaInferenceMemory) error {
	if memory.stream != nil {
		return memory.stream.Synchronize()
	}
	return cuda.Synchronize()
}

func (b *cudaInferenceBackend) Alloc(n int) (*gpu.Buffer, error) {
	if _, err := gpu.Elements(n); err != nil {
		return nil, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var native *cuda.Buffer
	err := inferenceCUDA(func() (err error) { native, err = cuda.Alloc(n * 4); return err })
	if err != nil {
		return nil, err
	}
	memory := &cudaInferenceMemory{buffer: native, stream: cuda.CurrentStream()}
	buffer, err := gpu.NewBuffer(b, n, memory, func() {
		// cuda.Free keys reuse by the calling thread's implicit stream. Return
		// this allocation to its own stream even after the caller switches it.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		previous := cuda.CurrentStream()
		cuda.SetCurrentStream(memory.stream)
		defer cuda.SetCurrentStream(previous)
		native.Free()
	})
	if err != nil {
		native.Free()
	}
	return buffer, err
}

func (b *cudaInferenceBackend) Upload(values []float32) (*gpu.Buffer, error) {
	buffer, err := b.Alloc(len(values))
	if err != nil {
		return nil, err
	}
	if err := b.CopyFrom(buffer, values); err != nil {
		b.Release(buffer)
		return nil, err
	}
	return buffer, nil
}

func (b *cudaInferenceBackend) Download(buffer *gpu.Buffer, n int) ([]float32, error) {
	if buffer == nil {
		return nil, gpu.ErrClosed
	}
	memory, err := b.memory(buffer, buffer.Len())
	if err != nil {
		return nil, err
	}
	if n < 0 || n > buffer.Len() {
		return nil, fmt.Errorf("%w: download length", gpu.ErrInvalidArgument)
	}
	values := make([]float32, n)
	err = withInferenceMemory(memory, func() error {
		if err := waitInferenceMemory(memory); err != nil {
			return err
		}
		return memory.buffer.CopyToHost(floatBytes(values))
	})
	return values, err
}

func (b *cudaInferenceBackend) CopyFrom(buffer *gpu.Buffer, values []float32) error {
	memory, err := b.memory(buffer, len(values))
	if err != nil {
		return err
	}
	return withInferenceMemory(memory, func() error {
		if err := waitInferenceMemory(memory); err != nil {
			return err
		}
		if err := memory.buffer.CopyFromHost(floatBytes(values)); err != nil {
			return err
		}
		// The runtime copy uses its synchronous/default-stream API. Complete
		// it before a non-blocking compute stream consumes the new values.
		return cuda.Synchronize()
	})
}

func (b *cudaInferenceBackend) Zero(buffer *gpu.Buffer) error {
	if buffer == nil {
		return gpu.ErrClosed
	}
	memory, err := b.memory(buffer, buffer.Len())
	if err != nil {
		return err
	}
	return withInferenceMemory(memory, func() error { return memory.buffer.Memset(0, memory.buffer.Size()) })
}

func (b *cudaInferenceBackend) Release(buffer *gpu.Buffer) {
	if buffer != nil && buffer.Backend() != b {
		panic(gpu.ErrBackendMismatch)
	}
	buffer.Close()
}

// Inference output allocations must not alias any input. This is portable to
// backends whose compiled operators do not support in-place execution.
func (b *cudaInferenceBackend) operands(out *gpu.Buffer, count int, inputs []*gpu.Buffer, sizes ...int) ([]*cuda.Buffer, error) {
	output, err := b.memory(out, count)
	if err != nil {
		return nil, err
	}
	memories := []*cudaInferenceMemory{output}
	for i, input := range inputs {
		if input == out {
			return nil, fmt.Errorf("%w: output aliases input", gpu.ErrInvalidArgument)
		}
		memory, err := b.memory(input, sizes[i])
		if err != nil {
			return nil, err
		}
		memories = append(memories, memory)
	}
	device, err := cuda.CurrentDevice()
	if err != nil {
		return nil, err
	}
	values := make([]*cuda.Buffer, len(memories))
	for i, memory := range memories {
		if memory.buffer.Device() != device || memory.stream != cuda.CurrentStream() {
			return nil, fmt.Errorf("%w: CUDA allocation device/stream differs from the current context", gpu.ErrBackendMismatch)
		}
		values[i] = memory.buffer
	}
	return values, nil
}

func (b *cudaInferenceBackend) MatMul(out, a, right *gpu.Buffer, m, n, k int, transA, transB bool) error {
	mn, err := gpu.Elements(m, n)
	if err != nil {
		return err
	}
	mk, err := gpu.Elements(m, k)
	if err != nil {
		return err
	}
	kn, err := gpu.Elements(k, n)
	if err != nil {
		return err
	}
	v, err := b.operands(out, mn, []*gpu.Buffer{a, right}, mk, kn)
	if err != nil {
		return err
	}
	if mn == 0 {
		return nil
	}
	if k == 0 {
		return b.Zero(out)
	}
	return inferenceCUDA(func() error {
		return (ExecutionOptions{}).gemmSingle(m, n, k, transA, transB, ptr(v[1]), ptr(v[2]), ptr(v[0]), 0)
	})
}

func (b *cudaInferenceBackend) binary(out, a, right *gpu.Buffer, count, kind int) error {
	v, err := b.operands(out, count, []*gpu.Buffer{a, right}, count, count)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		xp, bp, yp := ptr(v[1]), ptr(v[2]), ptr(v[0])
		n, op, one := int32(count), int32(kind), int32(1)
		launch("binary_f", count, unsafe.Pointer(&xp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&one), unsafe.Pointer(&n), unsafe.Pointer(&one), unsafe.Pointer(&one), unsafe.Pointer(&n), unsafe.Pointer(&one), unsafe.Pointer(&one), unsafe.Pointer(&n), unsafe.Pointer(&op))
		return nil
	})
}

func (b *cudaInferenceBackend) Add(out, a, right *gpu.Buffer, n int) error {
	return b.binary(out, a, right, n, 0)
}
func (b *cudaInferenceBackend) Mul(out, a, right *gpu.Buffer, n int) error {
	return b.binary(out, a, right, n, 2)
}

func (b *cudaInferenceBackend) MulScalar(out, a *gpu.Buffer, count int, scalar float32) error {
	v, err := b.operands(out, count, []*gpu.Buffer{a}, count)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		xp, yp := ptr(v[1]), ptr(v[0])
		n, kind := int32(count), int32(2)
		launch("scalar_f", count, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&scalar), unsafe.Pointer(&kind))
		return nil
	})
}

func (b *cudaInferenceBackend) GELU(out, a *gpu.Buffer, count int) error {
	v, err := b.operands(out, count, []*gpu.Buffer{a}, count)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		xp, yp := ptr(v[1]), ptr(v[0])
		n, kind := int32(count), int32(6)
		launch("unary_f", count, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&kind))
		return nil
	})
}

func (b *cudaInferenceBackend) Softmax(out, a *gpu.Buffer, rows, cols int) error {
	count, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 {
		return fmt.Errorf("%w: empty softmax axis", gpu.ErrInvalidArgument)
	}
	if _, err := gpu.Elements(rows, 32); err != nil {
		return err
	}
	v, err := b.operands(out, count, []*gpu.Buffer{a}, count)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		xp, yp := ptr(v[1]), ptr(v[0])
		r, c, inner, mode := int32(rows), int32(cols), int32(1), int32(0)
		launch("softmax_f", rows*32, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&r), unsafe.Pointer(&c), unsafe.Pointer(&inner), unsafe.Pointer(&mode))
		return nil
	})
}

func validInferenceEps(eps float32) bool {
	return eps > 0 && !math.IsInf(float64(eps), 0) && !math.IsNaN(float64(eps))
}

func (b *cudaInferenceBackend) LayerNorm(out, a, weight, bias *gpu.Buffer, rows, cols int, eps float32) error {
	count, err := gpu.Elements(rows, cols)
	if err != nil {
		return err
	}
	if cols == 0 || !validInferenceEps(eps) {
		return fmt.Errorf("%w: LayerNorm dimensions/epsilon", gpu.ErrInvalidArgument)
	}
	if _, err := gpu.Elements(rows, 32); err != nil {
		return err
	}
	v, err := b.operands(out, count, []*gpu.Buffer{a, weight, bias}, count, cols, cols)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		stats, err := cuda.Alloc(rows * 2 * 4)
		if err != nil {
			return err
		}
		defer stats.Free()
		xp, wp, bp, yp, sp := ptr(v[1]), ptr(v[2]), ptr(v[3]), ptr(v[0]), ptr(stats)
		r, c := int32(rows), int32(cols)
		launch("layernorm_f", rows*32, unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&sp), unsafe.Pointer(&r), unsafe.Pointer(&c), unsafe.Pointer(&eps))
		return nil
	})
}

func (b *cudaInferenceBackend) GroupNorm(out, a, weight, bias *gpu.Buffer, batch, frames, channels, groups int, eps float32) error {
	count, err := gpu.Elements(batch, frames, channels)
	if err != nil {
		return err
	}
	if frames == 0 || channels == 0 || groups < 1 || channels%groups != 0 || !validInferenceEps(eps) {
		return fmt.Errorf("%w: GroupNorm dimensions/epsilon", gpu.ErrInvalidArgument)
	}
	work, err := gpu.Elements(batch, groups, 256)
	if err != nil {
		return err
	}
	v, err := b.operands(out, count, []*gpu.Buffer{a, weight, bias}, count, channels, channels)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		stats, err := cuda.Alloc(batch * groups * 4 * 4)
		if err != nil {
			return err
		}
		defer stats.Free()
		xp, wp, bp, yp, sp := ptr(v[1]), ptr(v[2]), ptr(v[3]), ptr(v[0]), ptr(stats)
		n, bs, t, c, ng := int32(count), int32(batch), int32(frames), int32(channels), int32(groups)
		launch("group_stats", work, unsafe.Pointer(&xp), unsafe.Pointer(&sp), unsafe.Pointer(&bs), unsafe.Pointer(&t), unsafe.Pointer(&c), unsafe.Pointer(&ng), unsafe.Pointer(&eps))
		launch("group_f", count, unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&bp), unsafe.Pointer(&sp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&t), unsafe.Pointer(&c), unsafe.Pointer(&ng))
		return nil
	})
}

func (b *cudaInferenceBackend) Conv1d(out, x, weight, bias *gpu.Buffer, o gpu.Conv1dOptions) error {
	length, err := o.OutputLength()
	if err != nil {
		return err
	}
	xSize, err := gpu.Elements(o.Batch, o.InputLength, o.InChannels)
	if err != nil {
		return err
	}
	wSize, err := gpu.Elements(o.OutChannels, o.InChannels, o.Kernel)
	if err != nil {
		return err
	}
	ySize, err := gpu.Elements(o.Batch, length, o.OutChannels)
	if err != nil {
		return err
	}
	colSize, err := gpu.Elements(o.Batch, length, o.InChannels, o.Kernel)
	if err != nil {
		return err
	}
	inputs, sizes := []*gpu.Buffer{x, weight}, []int{xSize, wSize}
	if bias != nil {
		inputs, sizes = append(inputs, bias), append(sizes, o.OutChannels)
	}
	v, err := b.operands(out, ySize, inputs, sizes...)
	if err != nil {
		return err
	}
	return inferenceCUDA(func() error {
		cols, err := cuda.Alloc(colSize * 4)
		if err != nil {
			return err
		}
		defer cols.Free()
		if err := kernels.Im2colStride(v[1], cols, o.Batch, o.InChannels, o.InputLength, o.Kernel, length, o.Stride, o.Padding, o.Dilation); err != nil {
			return err
		}
		rows, depth := o.Batch*length, o.InChannels*o.Kernel
		if err := (ExecutionOptions{}).gemmSingle(rows, o.OutChannels, depth, false, true, ptr(cols), ptr(v[2]), ptr(v[0]), 0); err != nil {
			return err
		}
		if bias != nil {
			return kernels.AddBiasColumns(v[0], v[3], rows, o.OutChannels)
		}
		return nil
	})
}
