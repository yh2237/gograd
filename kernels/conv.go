package kernels

import (
	"fmt"
	"github.com/yh2237/gograd/cuda"
	"sync"
	"unsafe"
)

const convKernelSource = `
extern "C" __global__ void im2col1d_batched(const float* x, float* col, int batch, int channels, int length, int kernel, int dilation) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int per = channels * kernel * length;
	if (index >= batch * per) return;
	int t = index % length;
	int ck = (index / length) % (channels * kernel);
	int b = index / per;
	int k = ck % kernel;
	int c = ck / kernel;
	int src = t - dilation + k * dilation;
	float value = 0.0f;
	if (src >= 0 && src < length) value = x[(b * channels + c) * length + src];
	col[index] = value;
}
extern "C" __global__ void add_bias_batched(float* y, const float* bias, int batch, int outChannels, int length) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch * outChannels * length;
	if (index >= total) return;
	int o = (index / length) % outChannels;
	y[index] += bias[o];
}
`

var (
	convProgramOnce sync.Once
	convProgram     *cuda.Program
	im2colKernel    *cuda.Kernel
	biasKernel      *cuda.Kernel
	convProgramErr  error
)

func convKernels() (*cuda.Kernel, *cuda.Kernel, error) {
	convProgramOnce.Do(func() {
		program, err := cuda.Compile(convKernelSource)
		if err != nil {
			convProgramErr = err
			return
		}
		im2col, err := program.Function("im2col1d_batched")
		if err != nil {
			convProgramErr = err
			return
		}
		bias, err := program.Function("add_bias_batched")
		if err != nil {
			convProgramErr = err
			return
		}
		convProgram, im2colKernel, biasKernel = program, im2col, bias
	})
	return im2colKernel, biasKernel, convProgramErr
}

// Im2col gathers symmetric dilated patches from x [batch,channels,length] into
// col [batch,channels*kernel,length].
func Im2col(x, col *cuda.Buffer, batch, channels, length, kernel, dilation int) error {
	im2col, _, err := convKernels()
	if err != nil {
		return err
	}
	xAddr, colAddr := x.Pointer(), col.Pointer()
	b, c, t, k, d := int32(batch), int32(channels), int32(length), int32(kernel), int32(dilation)
	args := []unsafe.Pointer{
		unsafe.Pointer(&xAddr), unsafe.Pointer(&colAddr),
		unsafe.Pointer(&b), unsafe.Pointer(&c), unsafe.Pointer(&t), unsafe.Pointer(&k), unsafe.Pointer(&d),
	}
	return im2col.Launch(elementGrid(batch*channels*kernel*length), [3]int{256, 1, 1}, 0, nil, args)
}

// Conv1dForward computes a symmetric, dilated same-length convolution for
// row-major float32 tensors: x [batch,channels,length], weight
// [outChannels,channels,kernel], bias [outChannels], output
// [batch,outChannels,length]. One im2col kernel collects all batches, a
// strided-batched cuBLAS GEMM multiplies them, and one kernel adds the bias.
func Conv1dForward(blas *cuda.Blas, x, weight, bias, output *cuda.Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	return Conv1dForwardTo(blas, x, weight, bias, output, nil, batch, channels, length, outChannels, kernel, dilation)
}

// Conv1dForwardTo is Conv1dForward with an optional preallocated columns buffer.
// When columns is not nil the im2col result is kept there, so the weight
// gradient can reuse it instead of gathering patches again.
func Conv1dForwardTo(blas *cuda.Blas, x, weight, bias, output, columns *cuda.Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	_, addBias, err := convKernels()
	if err != nil {
		return err
	}
	if kernel < 1 || dilation < 1 {
		return fmt.Errorf("cuda: invalid conv kernel=%d dilation=%d", kernel, dilation)
	}
	if blas == nil {
		return fmt.Errorf("cuda: Conv1dForward requires a cuBLAS handle")
	}
	columnCount := channels * kernel
	if columns == nil {
		columns, err = cuda.Alloc(batch * columnCount * length * 4)
		if err != nil {
			return err
		}
		defer columns.Free()
	}

	if err := Im2col(x, columns, batch, channels, length, kernel, dilation); err != nil {
		return err
	}
	if err := blas.SgemmStridedBatchedRowMajor(batch, outChannels, length, columnCount, 1,
		weight.Pointer(), columnCount, 0,
		columns.Pointer(), length, int64(columnCount*length),
		0,
		output.Pointer(), length, int64(outChannels*length)); err != nil {
		return err
	}
	yAddr, biasAddr := output.Pointer(), bias.Pointer()
	b, o, l := int32(batch), int32(outChannels), int32(length)
	biasArgs := []unsafe.Pointer{unsafe.Pointer(&yAddr), unsafe.Pointer(&biasAddr), unsafe.Pointer(&b), unsafe.Pointer(&o), unsafe.Pointer(&l)}
	return addBias.Launch(elementGrid(batch*outChannels*length), [3]int{256, 1, 1}, 0, nil, biasArgs)
}
