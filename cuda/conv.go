package cuda

import (
	"fmt"
	"sync"
	"unsafe"
)

const convKernelSource = `
extern "C" __global__ void im2col1d(const float* x, float* col, int channels, int length, int kernel, int dilation) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int total = channels * kernel * length;
	if (index >= total) return;
	int t = index % length;
	int ck = index / length;
	int k = ck % kernel;
	int c = ck / kernel;
	int src = t - dilation + k * dilation;
	float value = 0.0f;
	if (src >= 0 && src < length) value = x[c * length + src];
	col[index] = value;
}
extern "C" __global__ void add_bias_row(float* y, const float* bias, int outChannels, int length) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= outChannels * length) return;
	int o = index / length;
	y[index] += bias[o];
}
`

var (
	convProgramOnce sync.Once
	convProgram     *Program
	im2colKernel    *Kernel
	biasKernel      *Kernel
	convProgramErr  error
)

func convKernels() (*Kernel, *Kernel, error) {
	convProgramOnce.Do(func() {
		program, err := Compile(convKernelSource)
		if err != nil {
			convProgramErr = err
			return
		}
		im2col, err := program.Function("im2col1d")
		if err != nil {
			convProgramErr = err
			return
		}
		bias, err := program.Function("add_bias_row")
		if err != nil {
			convProgramErr = err
			return
		}
		convProgram, im2colKernel, biasKernel = program, im2col, bias
	})
	return im2colKernel, biasKernel, convProgramErr
}

// Conv1dForward computes a symmetric, dilated same-length convolution for
// row-major float32 tensors: x [batch,channels,length], weight
// [outChannels,channels,kernel], bias [outChannels], output
// [batch,outChannels,length]. Each batch collects its patches with an im2col
// kernel and multiplies them with cuBLAS.
func Conv1dForward(blas *Blas, x, weight, bias, output *Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	im2col, addBias, err := convKernels()
	if err != nil {
		return err
	}
	if kernel < 1 || dilation < 1 {
		return fmt.Errorf("cuda: invalid conv kernel=%d dilation=%d", kernel, dilation)
	}
	if blas == nil {
		return fmt.Errorf("cuda: Conv1dForward requires a cuBLAS handle")
	}
	columns, err := Alloc(channels * kernel * length * 4)
	if err != nil {
		return err
	}
	defer columns.Free()

	columnCount := channels * kernel
	im2colGrid := [3]int{(columnCount*length + 255) / 256, 1, 1}
	biasGrid := [3]int{(outChannels*length + 255) / 256, 1, 1}
	for b := 0; b < batch; b++ {
		xOffset := x.Pointer() + uintptr(b*channels*length*4)
		yOffset := output.Pointer() + uintptr(b*outChannels*length*4)
		xAddr, colAddr := xOffset, columns.Pointer()
		c, t, k, d := int32(channels), int32(length), int32(kernel), int32(dilation)
		im2colArgs := []unsafe.Pointer{
			unsafe.Pointer(&xAddr), unsafe.Pointer(&colAddr),
			unsafe.Pointer(&c), unsafe.Pointer(&t), unsafe.Pointer(&k), unsafe.Pointer(&d),
		}
		if err := im2col.Launch(im2colGrid, [3]int{256, 1, 1}, 0, nil, im2colArgs); err != nil {
			return err
		}
		if err := blas.SgemmRowMajor(outChannels, length, columnCount, 1, weight.Pointer(), columnCount, columns.Pointer(), length, 0, yOffset, length); err != nil {
			return err
		}
		biasAddr := bias.Pointer()
		o, l := int32(outChannels), int32(length)
		biasArgs := []unsafe.Pointer{unsafe.Pointer(&yOffset), unsafe.Pointer(&biasAddr), unsafe.Pointer(&o), unsafe.Pointer(&l)}
		if err := addBias.Launch(biasGrid, [3]int{256, 1, 1}, 0, nil, biasArgs); err != nil {
			return err
		}
	}
	return nil
}
