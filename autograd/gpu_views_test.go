package autograd

import (
	"errors"
	"math/rand"
	"os"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	"github.com/yh2237/gograd/vulkan"
)

func (b *inferenceTestBackend) Gather(out *gpu.Buffer, src gpu.View) error {
	n, err := src.Validate(b)
	if err != nil {
		return err
	}
	x, err := b.memory(src.Buffer, src.Buffer.Len())
	if err != nil {
		return err
	}
	y, err := b.memory(out, n)
	if err != nil {
		return err
	}
	for i := range y.data {
		y.data[i] = x.data[storageIndex(i, src.Shape, src.Strides, src.Offset)]
	}
	return nil
}
func (b *inferenceTestBackend) Scatter(dst gpu.View, src *gpu.Buffer) error {
	n, err := dst.ValidateWrite(b)
	if err != nil {
		return err
	}
	x, err := b.memory(src, n)
	if err != nil {
		return err
	}
	y, err := b.memory(dst.Buffer, dst.Buffer.Len())
	if err != nil {
		return err
	}
	for i, value := range x.data {
		y.data[storageIndex(i, dst.Shape, dst.Strides, dst.Offset)] = value
	}
	return nil
}
func (b *inferenceTestBackend) MatMulViews(out *gpu.Buffer, a, right gpu.View) error {
	if _, err := a.Validate(b); err != nil {
		return err
	}
	if _, err := right.Validate(b); err != nil {
		return err
	}
	shape, err := gpu.MatMulShape(a.Shape, right.Shape)
	if err != nil {
		return err
	}
	count, err := gpu.Elements(shape...)
	if err != nil {
		return err
	}
	x, err := b.memory(a.Buffer, a.Buffer.Len())
	if err != nil {
		return err
	}
	w, err := b.memory(right.Buffer, right.Buffer.Len())
	if err != nil {
		return err
	}
	y, err := b.memory(out, count)
	if err != nil {
		return err
	}
	m, k, n := a.Shape[len(a.Shape)-2], a.Shape[len(a.Shape)-1], right.Shape[len(right.Shape)-1]
	batch := shape[:len(shape)-2]
	batches, _ := gpu.Elements(batch...)
	batchOffset := func(z int, v gpu.View) int {
		offset, rest, pad := v.Offset, z, len(batch)-len(v.Shape)+2
		for i := len(batch) - 1; i >= 0; i-- {
			coord := rest % batch[i]
			rest /= batch[i]
			if j := i - pad; j >= 0 && v.Shape[j] != 1 {
				offset += coord * v.Strides[j]
			}
		}
		return offset
	}
	for z := 0; z < batches; z++ {
		ao, bo := batchOffset(z, a), batchOffset(z, right)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum float32
				for d := 0; d < k; d++ {
					sum += x.data[ao+i*a.Strides[len(a.Shape)-2]+d*a.Strides[len(a.Shape)-1]] * w.data[bo+d*right.Strides[len(right.Shape)-2]+j*right.Strides[len(right.Shape)-1]]
				}
				y.data[(z*m+i)*n+j] = sum
			}
		}
	}
	return nil
}

func inferenceTestDevices(t *testing.T, fn func(*testing.T, tensor.Device)) {
	t.Run("fixture", func(t *testing.T) { fn(t, fixtureDevice) })
	t.Run("vulkan", func(t *testing.T) {
		if !vulkan.Available() {
			if os.Getenv("GOGRAD_REQUIRE_VULKAN") == "1" {
				t.Fatal(vulkan.Err())
			}
			t.Skipf("Vulkan unavailable: %v", vulkan.Err())
		}
		fn(t, tensor.Vulkan)
	})
}

func TestInferenceGPUComposedViews(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		x, err := New(data, []int{2, 3, 4}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		_, allocations, _ := inferenceFixture.snapshot()
		v := Slice(Permute(Slice(x, 2, 1, 4), 2, 0, 1), 2, 1, 3)
		defer v.ReleaseGraph()
		if v.storage != x.storage || v.Offset != 5 {
			t.Fatal("view composition copied storage or lost offset")
		}
		if device == fixtureDevice {
			if _, after, _ := inferenceFixture.snapshot(); after != allocations {
				t.Fatal("metadata-only view allocated GPU memory")
			}
		}
		want := make([]float32, v.Numel())
		for i := range want {
			want[i] = data[storageIndex(i, v.Shape, v.Strides, v.Offset)]
		}
		host, err := v.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "composed view", host, want, 0, 0)
		updates := make([]float32, v.Numel())
		for i := range updates {
			updates[i] = float32(100 + i)
			data[storageIndex(i, v.Shape, v.Strides, v.Offset)] = updates[i]
		}
		if err := v.CopyFrom(updates); err != nil {
			t.Fatal(err)
		}
		if x.storage.version.Load() != 1 {
			t.Fatal("view write did not invalidate shared storage")
		}
		host, err = x.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "base after scatter", host, data, 0, 0)
		flat := Reshape(v, v.Numel())
		defer flat.ReleaseGraph()
		if flat.storage == x.storage {
			t.Fatal("noncompact reshape did not materialize")
		}
		host, err = flat.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "noncompact reshape", host, updates, 0, 0)
		detached := v.Detach()
		defer detached.Close()
		v.ReleaseGraph()
		host, err = detached.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "noncompact detach", host, updates, 0, 0)
		prefix := Slice(x, 0, 1, 2)
		defer prefix.ReleaseGraph()
		reshaped := Reshape(prefix, 12)
		defer reshaped.ReleaseGraph()
		if reshaped.storage != x.storage || reshaped.Offset != 12 {
			t.Fatal("compact offset reshape copied or lost offset")
		}
		empty := Slice(prefix, 0, 1, 1)
		defer empty.ReleaseGraph()
		host, err = empty.ToHost()
		if err != nil || len(host) != 0 {
			t.Fatalf("empty view: %v %v", host, err)
		}
	})
}

func TestInferenceGPUExpandAndBroadcast(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		x, err := New([]float32{1, 2, 3}, []int{1, 3}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		v := Expand(x, 2, 4, 3)
		defer v.ReleaseGraph()
		values := make([]float32, v.Numel())
		for i := range values {
			values[i] = float32(i%3 + 1)
		}
		host, err := v.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "expanded", host, values, 0, 0)
		if err := v.CopyFrom(values); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
		if x.storage.version.Load() != 0 {
			t.Fatal("rejected overlapping write changed version")
		}
		bias, err := New([]float32{10, 20, 30}, []int{3}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer bias.Close()
		_, _, downloads := inferenceFixture.snapshot()
		y := MulScalar(Add(v, bias), .5)
		defer y.ReleaseGraph()
		if device == fixtureDevice {
			if _, _, after := inferenceFixture.snapshot(); after != downloads {
				t.Fatal("broadcast downloaded values")
			}
		}
		host, err = y.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		for i := range values {
			values[i] = float32((i%3+1)*11) * .5
		}
		conv2dClose(t, "broadcast", host, values, 0, 0)
		z, err := Zeros([]int{0, 3}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer z.Close()
		zero := Add(z, bias)
		defer zero.ReleaseGraph()
		host, err = zero.ToHost()
		if err != nil || len(host) != 0 || zero.Shape[0] != 0 {
			t.Fatalf("zero broadcast: %v %v", host, err)
		}
	})
}

func TestInferenceGPUViewLifetime(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		var before uint64
		if device == tensor.Vulkan {
			before = vulkan.MemoryStats().LiveBytes
		}
		x, err := New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, false)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		v := Transpose(x, 0, 1)
		defer v.Close()
		v.RetainData()
		x.Close()
		v.ReleaseGraph()
		host, err := v.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "retained transpose", host, []float32{1, 4, 2, 5, 3, 6}, 0, 0)
		v.Close()
		if device == tensor.Vulkan && vulkan.MemoryStats().LiveBytes != before {
			t.Fatal("view allocation lease leaked")
		}
	})
}

func TestInferenceGPULinear(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		c, cc := NewExecutionContext(), NewExecutionContext()
		layer, err := NewLinearLayer(5, 7, device, rand.New(rand.NewSource(17)))
		if err != nil {
			t.Fatal(err)
		}
		defer layer.Weight.Close()
		defer layer.Bias.Close()
		cpu, err := NewLinearLayer(5, 7, tensor.CPU, rand.New(rand.NewSource(17)))
		if err != nil {
			t.Fatal(err)
		}
		defer cpu.Weight.Close()
		defer cpu.Bias.Close()
		if err := c.BindModule(&layer.Module); err != nil {
			t.Fatal(err)
		}
		if err := cc.BindModule(&cpu.Module); err != nil {
			t.Fatal(err)
		}
		data := make([]float32, 30)
		for i := range data {
			data[i] = float32(i%13-6) / 8
		}
		x, err := c.New(data, []int{2, 3, 5}, device, true)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		cx, err := cc.New(data, []int{2, 3, 5}, tensor.CPU, true)
		if err != nil {
			t.Fatal(err)
		}
		defer cx.Close()
		_, _, downloads := inferenceFixture.snapshot()
		var y, ref *Tensor
		c.NoGrad(func() { y = layer.Forward(x) })
		defer y.ReleaseGraph()
		cc.NoGrad(func() { ref = cpu.Forward(cx) })
		defer ref.ReleaseGraph()
		if device == fixtureDevice {
			if _, _, after := inferenceFixture.snapshot(); after != downloads {
				t.Fatal("linear projection downloaded inputs")
			}
		}
		x.Close()
		layer.Weight.Close()
		layer.Bias.Close()
		host, err := y.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "linear", host, ref.Data, 1e-4, 1e-4)
	})
}

func TestInferenceGPUBatchedMatMul(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		c, cc := NewExecutionContext(), NewExecutionContext()
		a, right := make([]float32, 30), make([]float32, 140)
		for i := range a {
			a[i] = float32(i%17-8) / 16
		}
		for i := range right {
			right[i] = float32(i%13-6) / 8
		}
		x, err := c.New(a, []int{2, 1, 3, 5}, device, true)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		w, err := c.New(right, []int{1, 4, 7, 5}, device, true)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		cx, err := cc.New(a, []int{2, 1, 3, 5}, tensor.CPU, true)
		if err != nil {
			t.Fatal(err)
		}
		defer cx.Close()
		cw, err := cc.New(right, []int{1, 4, 7, 5}, tensor.CPU, true)
		if err != nil {
			t.Fatal(err)
		}
		defer cw.Close()
		_, allocations, downloads := inferenceFixture.snapshot()
		var y, ref *Tensor
		c.NoGrad(func() { y = MatMul(x, Transpose(w, 2, 3)) })
		defer y.ReleaseGraph()
		cc.NoGrad(func() { ref = MatMul(cx, Transpose(cw, 2, 3)) })
		defer ref.ReleaseGraph()
		if device == fixtureDevice {
			if _, after, reads := inferenceFixture.snapshot(); after != allocations+1 || reads != downloads {
				t.Fatal("view GEMM materialized or downloaded operands")
			}
		}
		host, err := y.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "batched broadcast matmul", host, ref.Data, 1e-4, 1e-4)
	})
}

func TestInferenceGPUViewDispatchFailure(t *testing.T) {
	x, err := New([]float32{1, 2, 3}, []int{1, 3}, fixtureDevice, false)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	v := Expand(x, 4, 3)
	defer v.ReleaseGraph()
	live, _, _ := inferenceFixture.snapshot()
	inferenceFixture.mu.Lock()
	inferenceFixture.failNext = true
	inferenceFixture.mu.Unlock()
	assertGPUErrorPanic(t, injectedInferenceError, func() { Add(v, v) })
	if after, _, _ := inferenceFixture.snapshot(); after != live {
		t.Fatal("failed broadcast leaked scratch or output")
	}
}
