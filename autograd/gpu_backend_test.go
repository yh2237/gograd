package autograd

import (
	"errors"
	"fmt"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func TestGPUDeviceRolloutGuards(t *testing.T) {
	for _, device := range []tensor.Device{tensor.Vulkan, tensor.DirectML} {
		if !gpu.Available(string(device)) {
			if _, err := New([]float32{1}, []int{1}, device, false); !errors.Is(err, gpu.ErrUnavailable) {
				t.Fatal(err)
			}
			if _, err := Zeros([]int{1}, device, false); !errors.Is(err, gpu.ErrUnavailable) {
				t.Fatal(err)
			}
		}
		if _, err := tensor.NewOn(device, 1); !errors.Is(err, gpu.ErrUnavailable) {
			t.Fatal(err)
		}
		// Exercise training rejection independently of a native driver. Changing
		// Device manually is not a supported way to move a real tensor.
		x := Must([]float32{1}, []int{1}, true)
		x.Device = device
		params := []Parameter{{Name: "weight", Value: x}}
		if err := x.Backward(); !errors.Is(err, gpu.ErrInferenceOnly) {
			t.Fatal(err)
		}
		if x.Grad != nil {
			t.Fatal("backward changed gradients before rejection")
		}
		if _, err := NewSGD(params, SGDOptions{LR: .1}); !errors.Is(err, gpu.ErrInferenceOnly) {
			t.Fatal(err)
		}
		if _, err := NewAdamWGroups(params, .1, []ParameterGroupSpec{{Name: "all", Params: params}}); !errors.Is(err, gpu.ErrInferenceOnly) {
			t.Fatal(err)
		}
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { NewAdamW(params, .1, 0) })
		opt := &AdamW{Params: params, StepCount: 7}
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, opt.Step)
		if opt.StepCount != 7 {
			t.Fatal("optimizer advanced before rejection")
		}
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { ClipGradNorm(params, 1) })
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { Add(x, x) })
		x.Close()
	}
}

func assertGPUErrorPanic(t *testing.T, target error, fn func()) {
	t.Helper()
	defer func() {
		value := recover()
		err, ok := value.(error)
		if !ok || !errors.Is(err, target) {
			t.Fatalf("panic %v, want %v", value, target)
		}
	}()
	fn()
}

func TestCUDAInferenceBackend(t *testing.T) {
	if !gpu.Available("cuda") {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ctx.Close() })
	b, err := gpu.Get("cuda")
	if err != nil {
		t.Fatal(err)
	}
	before := cuda.MemoryStats().LiveBytes
	values := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(i%37-18) / 32
		}
		return v
	}
	upload := func(t *testing.T, v []float32) *gpu.Buffer {
		t.Helper()
		x, err := b.Upload(v)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { b.Release(x) })
		return x
	}
	alloc := func(t *testing.T, n int) *gpu.Buffer {
		t.Helper()
		x, err := b.Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { b.Release(x) })
		return x
	}
	check := func(t *testing.T, out *gpu.Buffer, want []float32, tol float64) {
		t.Helper()
		got, err := b.Download(out, out.Len())
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "backend output", got, want, tol, tol)
	}
	t.Run("memory", func(t *testing.T) {
		x := upload(t, []float32{1, 2, 3})
		check(t, x, []float32{1, 2, 3}, 0)
		if prefix, err := b.Download(x, 2); err != nil || len(prefix) != 2 || prefix[1] != 2 {
			t.Fatalf("prefix %v %v", prefix, err)
		}
		if err := b.CopyFrom(x, []float32{4, 5, 6}); err != nil {
			t.Fatal(err)
		}
		check(t, x, []float32{4, 5, 6}, 0)
		if err := b.Zero(x); err != nil {
			t.Fatal(err)
		}
		check(t, x, []float32{0, 0, 0}, 0)
		if err := b.CopyFrom(x, []float32{1}); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
		if _, err := b.Download(x, 4); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
		if err := b.Add(x, x, x, 3); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
		foreign, err := (&cudaInferenceBackend{}).Upload([]float32{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(foreign.Close)
		if err := b.Add(alloc(t, 3), x, foreign, 3); !errors.Is(err, gpu.ErrBackendMismatch) {
			t.Fatal(err)
		}
		assertGPUErrorPanic(t, gpu.ErrBackendMismatch, func() { b.Release(foreign) })
		b.Release(x)
		b.Release(x)
		if _, err := b.Download(x, 3); !errors.Is(err, gpu.ErrClosed) {
			t.Fatal(err)
		}
		if _, err := b.Alloc(-1); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
		empty := upload(t, nil)
		check(t, empty, nil, 0)
	})
	t.Run("streamOwnership", func(t *testing.T) {
		first, err := cuda.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { first.Destroy() })
		second, err := cuda.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { second.Destroy() })
		previous := cuda.CurrentStream()
		defer cuda.SetCurrentStream(previous)
		cuda.SetCurrentStream(first)
		x, out := upload(t, []float32{1, 2, 3}), alloc(t, 3)
		if err := b.MulScalar(out, x, 3, 2); err != nil {
			t.Fatal(err)
		}
		cuda.SetCurrentStream(second)
		if err := b.MulScalar(out, x, 3, 3); !errors.Is(err, gpu.ErrBackendMismatch) {
			t.Fatal(err)
		}
		check(t, out, []float32{2, 4, 6}, 0)
		if cuda.CurrentStream() != second {
			t.Fatal("download changed caller stream")
		}
		if err := b.CopyFrom(x, []float32{4, 5, 6}); err != nil {
			t.Fatal(err)
		}
		if err := b.Zero(out); err != nil {
			t.Fatal(err)
		}
		if cuda.CurrentStream() != second {
			t.Fatal("transfer changed caller stream")
		}
		cuda.SetCurrentStream(first)
		if err := b.Add(out, x, x, 3); err != nil {
			t.Fatal(err)
		}
		check(t, out, []float32{8, 10, 12}, 0)
		memory, err := b.(*cudaInferenceBackend).memory(out, 3)
		if err != nil {
			t.Fatal(err)
		}
		pointer := memory.buffer.Pointer()
		cuda.SetCurrentStream(second)
		b.Release(out)
		other := alloc(t, 3)
		otherMemory, err := b.(*cudaInferenceBackend).memory(other, 3)
		if err != nil {
			t.Fatal(err)
		}
		if otherMemory.buffer.Pointer() == pointer {
			t.Fatal("allocation reused on wrong stream")
		}
		cuda.SetCurrentStream(first)
		reused := alloc(t, 3)
		reusedMemory, err := b.(*cudaInferenceBackend).memory(reused, 3)
		if err != nil {
			t.Fatal(err)
		}
		if reusedMemory.buffer.Pointer() != pointer {
			t.Fatal("allocation not returned to its own stream")
		}
	})
	for _, dims := range [][3]int{{7, 11, 13}, {33, 17, 257}, {3, 5, 0}, {0, 5, 3}} {
		for _, ta := range []bool{false, true} {
			for _, tb := range []bool{false, true} {
				t.Run(fmt.Sprintf("matmul/%v/%t/%t", dims, ta, tb), func(t *testing.T) {
					m, n, k := dims[0], dims[1], dims[2]
					a, right, want := values(m*k), values(k*n), make([]float32, m*n)
					tensor.SGEMMOp(want, a, right, m, n, k, ta, tb)
					out := alloc(t, m*n)
					if err := b.MatMul(out, upload(t, a), upload(t, right), m, n, k, ta, tb); err != nil {
						t.Fatal(err)
					}
					check(t, out, want, 1e-4)
				})
			}
		}
	}
	t.Run("pointwise", func(t *testing.T) {
		const n = 513
		a, right := values(n), values(n)
		x, y, out := upload(t, a), upload(t, right), alloc(t, n)
		want := make([]float32, n)
		if err := b.Add(out, x, y, n); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			want[i] = a[i] + right[i]
		}
		check(t, out, want, 0)
		if err := b.Mul(out, x, y, n); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			want[i] = a[i] * right[i]
		}
		check(t, out, want, 0)
		if err := b.MulScalar(out, x, n, .375); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			want[i] = a[i] * .375
		}
		check(t, out, want, 0)
		if err := b.GELU(out, x, n); err != nil {
			t.Fatal(err)
		}
		cpu := GELU(conv2dTestTensor(t, a, []int{n}, tensor.CPU, false), true)
		defer cpu.ReleaseGraph()
		check(t, out, cpu.Data, 1e-5)
	})
	t.Run("softmax", func(t *testing.T) {
		const rows, cols = 7, 33
		data, out := values(rows*cols), alloc(t, rows*cols)
		if err := b.Softmax(out, upload(t, data), rows, cols); err != nil {
			t.Fatal(err)
		}
		cpu := Softmax(conv2dTestTensor(t, data, []int{rows, cols}, tensor.CPU, false), -1)
		defer cpu.ReleaseGraph()
		check(t, out, cpu.Data, 1e-5)
		if err := b.Softmax(out, out, rows, 0); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
	})
	t.Run("layernorm", func(t *testing.T) {
		const rows, cols = 13, 17
		data, weight, bias := values(rows*cols), values(cols), values(cols)
		out := alloc(t, len(data))
		if err := b.LayerNorm(out, upload(t, data), upload(t, weight), upload(t, bias), rows, cols, 1e-5); err != nil {
			t.Fatal(err)
		}
		cpu := LayerNorm(conv2dTestTensor(t, data, []int{rows, cols}, tensor.CPU, false), conv2dTestTensor(t, weight, []int{cols}, tensor.CPU, false), conv2dTestTensor(t, bias, []int{cols}, tensor.CPU, false), 1e-5)
		defer cpu.ReleaseGraph()
		check(t, out, cpu.Data, 1e-4)
	})
	t.Run("groupnorm", func(t *testing.T) {
		const batch, frames, channels, groups = 2, 7, 12, 3
		data, weight, bias := values(batch*frames*channels), values(channels), values(channels)
		out := alloc(t, len(data))
		if err := b.GroupNorm(out, upload(t, data), upload(t, weight), upload(t, bias), batch, frames, channels, groups, 1e-5); err != nil {
			t.Fatal(err)
		}
		cpu := GroupNorm(conv2dTestTensor(t, data, []int{batch, frames, channels}, tensor.CPU, false), conv2dTestTensor(t, weight, []int{channels}, tensor.CPU, false), conv2dTestTensor(t, bias, []int{channels}, tensor.CPU, false), groups, 1e-5)
		defer cpu.ReleaseGraph()
		check(t, out, cpu.Data, 1e-4)
	})
	for _, withBias := range []bool{false, true} {
		t.Run(fmt.Sprintf("conv1d/bias%t", withBias), func(t *testing.T) {
			o := gpu.Conv1dOptions{Batch: 2, InputLength: 11, InChannels: 3, OutChannels: 5, Kernel: 4, Stride: 2, Padding: 3, Dilation: 2}
			length, err := o.OutputLength()
			if err != nil {
				t.Fatal(err)
			}
			data, weight, bias := values(o.Batch*o.InputLength*o.InChannels), values(o.OutChannels*o.InChannels*o.Kernel), make([]float32, o.OutChannels)
			var biasBuffer *gpu.Buffer
			if withBias {
				bias = values(o.OutChannels)
				biasBuffer = upload(t, bias)
			}
			out := alloc(t, o.Batch*length*o.OutChannels)
			if err := b.Conv1d(out, upload(t, data), upload(t, weight), biasBuffer, o); err != nil {
				t.Fatal(err)
			}
			cpu := Conv1dGEMMStride(conv2dTestTensor(t, data, []int{o.Batch, o.InputLength, o.InChannels}, tensor.CPU, false), conv2dTestTensor(t, weight, []int{o.OutChannels, o.InChannels, o.Kernel}, tensor.CPU, false), conv2dTestTensor(t, bias, []int{o.OutChannels}, tensor.CPU, false), o.Stride, o.Padding, o.Dilation)
			defer cpu.ReleaseGraph()
			check(t, out, cpu.Data, 1e-4)
		})
	}
	if after := cuda.MemoryStats().LiveBytes; after != before {
		t.Fatalf("live CUDA memory leaked: %d -> %d", before, after)
	}
}
