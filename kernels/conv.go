package kernels

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// The convolution works on time-major tensors. im2col lays patches out as
// [channels*kernel, batch*length] so every batch folds into one large GEMM
// column, which removes the per-utterance loop and all layout transposes.
const convKernelSource = `
extern "C" __global__ void im2col1d_batched(const float* x, float* col, int batch, int channels, int length, int kernel, int dilation) {
	// col is [batch*length, channels*kernel]; x is [batch,length,channels].
	// Consecutive threads walk the channel-tap axis, so both the read of x and
	// the write of col are contiguous.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int per = channels * kernel;
	if (index >= batch * length * per) return;
	int ck = index % per;
	int bt = index / per;
	int t = bt % length;
	int b = bt / length;
	int k = ck % kernel;
	int c = ck / kernel;
	int src = t - (kernel / 2) * dilation + k * dilation;
	float value = 0.0f;
	if (src >= 0 && src < length) value = x[(b * length + src) * channels + c];
	col[index] = value;
}
extern "C" __global__ void col2im1d(const float* col, float* dx, int batch, int channels, int length, int kernel, int dilation) {
	// col is [batch*length, channels*kernel]; dx is [batch,length,channels].
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= batch * length * channels) return;
	int c = index % channels;
	int bt = index / channels;
	int t = bt % length;
	int b = bt / length;
	float sum = 0.0f;
	int ck = c * kernel;
	int stride = channels * kernel;
	for (int k = 0; k < kernel; k++) {
		int src = t + (kernel / 2) * dilation - k * dilation;
		if (src < 0 || src >= length) continue;
		sum += col[(b * length + src) * stride + ck + k];
	}
	dx[index] += sum;
}
extern "C" __global__ void im2col1d_stride(const float* x, float* col, int batch, int channels, int length, int kernel, int out_length, int stride, int padding, int dilation) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int per = channels * kernel;
	if (index >= batch * out_length * per) return;
	int ck = index % per;
	int bt = index / per;
	int at = bt % out_length;
	int b = bt / out_length;
	int src = at * stride - padding + (ck % kernel) * dilation;
	col[index] = src >= 0 && src < length ? x[(b * length + src) * channels + ck / kernel] : 0.0f;
}
extern "C" __global__ void col2im1d_stride(const float* col, float* dx, int batch, int channels, int length, int kernel, int out_length, int stride, int padding, int dilation) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= batch * length * channels) return;
	int c = index % channels;
	int bt = index / channels;
	int t = bt % length;
	int b = bt / length;
	float sum = 0.0f;
	for (int tap = 0; tap < kernel; tap++) {
		int at_times_stride = t + padding - tap * dilation;
		if (at_times_stride < 0 || at_times_stride % stride != 0) continue;
		int at = at_times_stride / stride;
		if (at < out_length) sum += col[((b * out_length + at) * channels + c) * kernel + tap];
	}
	dx[index] += sum;
}
`

type convKernelSet struct {
	im2col       *cuda.Kernel
	col2im       *cuda.Kernel
	im2colStride *cuda.Kernel
	col2imStride *cuda.Kernel
}

var convCache cuda.DeviceCache[*convKernelSet]

func convKernels() (*convKernelSet, error) {
	return convCache.Get(func() (*convKernelSet, error) {
		program, err := cuda.Compile(convKernelSource)
		if err != nil {
			return nil, err
		}
		set := &convKernelSet{}
		for name, target := range map[string]**cuda.Kernel{
			"im2col1d_batched": &set.im2col,
			"col2im1d":         &set.col2im,
			"im2col1d_stride":  &set.im2colStride,
			"col2im1d_stride":  &set.col2imStride,
		} {
			kernel, err := program.Function(name)
			if err != nil {
				_ = program.Close()
				return nil, err
			}
			*target = kernel
		}
		return set, nil
	})
}

// Im2col gathers symmetric dilated patches from time-major x [batch,length,
// channels] into col [channels*kernel, batch*length].
func Im2col(x, col *cuda.Buffer, batch, channels, length, kernel, dilation int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	xAddr, colAddr := x.Pointer(), col.Pointer()
	b, c, t, k, d := int32(batch), int32(channels), int32(length), int32(kernel), int32(dilation)
	args := []unsafe.Pointer{
		unsafe.Pointer(&xAddr), unsafe.Pointer(&colAddr),
		unsafe.Pointer(&b), unsafe.Pointer(&c), unsafe.Pointer(&t), unsafe.Pointer(&k), unsafe.Pointer(&d),
	}
	return set.im2col.Launch(elementGrid(batch*channels*kernel*length), [3]int{256, 1, 1}, 0, nil, args)
}

// Col2im scatters a [channels*kernel, batch*length] patch gradient into the
// time-major input gradient [batch,length,channels].
func Col2im(col, dx *cuda.Buffer, batch, channels, length, kernel, dilation int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	colAddr, dxAddr := col.Pointer(), dx.Pointer()
	b, c, t, k, d := int32(batch), int32(channels), int32(length), int32(kernel), int32(dilation)
	args := []unsafe.Pointer{unsafe.Pointer(&colAddr), unsafe.Pointer(&dxAddr),
		unsafe.Pointer(&b), unsafe.Pointer(&c), unsafe.Pointer(&t), unsafe.Pointer(&k), unsafe.Pointer(&d)}
	return set.col2im.Launch(elementGrid(batch*channels*length), [3]int{256, 1, 1}, 0, nil, args)
}

// Im2colStride gathers patches for an explicit stride, padding and dilation.
func Im2colStride(x, col *cuda.Buffer, batch, channels, length, kernel, outLength, stride, padding, dilation int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	xAddr, colAddr := x.Pointer(), col.Pointer()
	args := int32Args(int32(batch), int32(channels), int32(length), int32(kernel), int32(outLength), int32(stride), int32(padding), int32(dilation))
	args = append([]unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&colAddr)}, args...)
	return set.im2colStride.Launch(elementGrid(batch*channels*kernel*outLength), [3]int{256, 1, 1}, 0, nil, args)
}

// Col2imStride gathers patch gradients into the input gradient.
func Col2imStride(col, dx *cuda.Buffer, batch, channels, length, kernel, outLength, stride, padding, dilation int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	colAddr, dxAddr := col.Pointer(), dx.Pointer()
	args := int32Args(int32(batch), int32(channels), int32(length), int32(kernel), int32(outLength), int32(stride), int32(padding), int32(dilation))
	args = append([]unsafe.Pointer{unsafe.Pointer(&colAddr), unsafe.Pointer(&dxAddr)}, args...)
	return set.col2imStride.Launch(elementGrid(batch*channels*length), [3]int{256, 1, 1}, 0, nil, args)
}
