package kernels

import (
	"fmt"
	"sync"
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
extern "C" __global__ void transpose_add_tanh(const float* conv, const float* state, const float* bias, float* out, int batch, int channels, int length) {
	// conv is [channels, batch*length]; state, out are [batch,length,channels].
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= batch * length * channels) return;
	int c = index % channels;
	int bt = index / channels;
	int t = bt % length;
	int b = bt / length;
	float value = state[(b * length + t) * channels + c] + conv[c * batch * length + b * length + t] + bias[c];
	out[index] = tanhf(value);
}
extern "C" __global__ void to_hbt(const float* in, float* out, int batch, int channels, int length) {
	// in is [batch,length,channels]; out is [channels, batch*length].
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= channels * batch * length) return;
	int t = index % length;
	int tmp = index / length;
	int b = tmp % batch;
	int h = tmp / batch;
	out[index] = in[(b * length + t) * channels + h];
}
extern "C" __global__ void row_sum(const float* x, float* out, int rows, int cols, int chunks) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= rows * chunks) return;
	int chunk = index % chunks;
	int r = index / chunks;
	int begin = (cols * chunk) / chunks;
	int end = (cols * (chunk + 1)) / chunks;
	float sum = 0.0f;
	for (int c = begin; c < end; c++) sum += x[r * cols + c];
	if (sum != 0.0f) atomicAdd(&out[r], sum);
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
`

type convKernelSet struct {
	im2col           *cuda.Kernel
	transposeAddTanh *cuda.Kernel
	toHBT            *cuda.Kernel
	rowSum           *cuda.Kernel
	col2im           *cuda.Kernel
}

var (
	convProgramOnce sync.Once
	convSet         *convKernelSet
	convProgramErr  error
)

func convKernels() (*convKernelSet, error) {
	convProgramOnce.Do(func() {
		program, err := cuda.Compile(convKernelSource)
		if err != nil {
			convProgramErr = err
			return
		}
		set := &convKernelSet{}
		for name, target := range map[string]**cuda.Kernel{
			"im2col1d_batched":   &set.im2col,
			"transpose_add_tanh": &set.transposeAddTanh,
			"to_hbt":             &set.toHBT,
			"row_sum":            &set.rowSum,
			"col2im1d":           &set.col2im,
		} {
			kernel, err := program.Function(name)
			if err != nil {
				convProgramErr = err
				return
			}
			*target = kernel
		}
		convSet = set
	})
	return convSet, convProgramErr
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

// TransposeAddTanh writes tanh(state + transpose(conv) + bias), where conv is
// [channels,batch*length] and state and out are [batch,length,channels].
func TransposeAddTanh(conv, state, bias, out *cuda.Buffer, batch, channels, length int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	convAddr, stateAddr, biasAddr, outAddr := conv.Pointer(), state.Pointer(), bias.Pointer(), out.Pointer()
	b, c, l := int32(batch), int32(channels), int32(length)
	args := []unsafe.Pointer{unsafe.Pointer(&convAddr), unsafe.Pointer(&stateAddr), unsafe.Pointer(&biasAddr), unsafe.Pointer(&outAddr),
		unsafe.Pointer(&b), unsafe.Pointer(&c), unsafe.Pointer(&l)}
	return set.transposeAddTanh.Launch(elementGrid(batch*channels*length), [3]int{256, 1, 1}, 0, nil, args)
}

// ToHBT permutes a time-major [batch,length,channels] tensor into
// [channels, batch*length], the layout the convolution gradients expect.
func ToHBT(in, out *cuda.Buffer, batch, channels, length int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	inAddr, outAddr := in.Pointer(), out.Pointer()
	b, c, l := int32(batch), int32(channels), int32(length)
	args := []unsafe.Pointer{unsafe.Pointer(&inAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&b), unsafe.Pointer(&c), unsafe.Pointer(&l)}
	return set.toHBT.Launch(elementGrid(channels*batch*length), [3]int{256, 1, 1}, 0, nil, args)
}

// RowSum accumulates out[r] += sum_c x[r,c] for a [rows,cols] matrix. out must
// be zero before the call.
func RowSum(x, out *cuda.Buffer, rows, cols int) error {
	set, err := convKernels()
	if err != nil {
		return err
	}
	chunks := 1
	if cols > 1 {
		chunks = 256
		if chunks > cols {
			chunks = cols
		}
	}
	xAddr, outAddr := x.Pointer(), out.Pointer()
	args := int32Args(int32(rows), int32(cols), int32(chunks))
	args = append([]unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&outAddr)}, args...)
	return set.rowSum.Launch(elementGrid(rows*chunks), [3]int{256, 1, 1}, 0, nil, args)
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

// Conv1dForwardTo gathers time-major x [batch,length,channels] into columns
// [channels*kernel, batch*length] and computes conv [outChannels, batch*length]
// = weight*columns with one GEMM. No bias or activation is applied here.
func Conv1dForwardTo(blas *cuda.Blas, x, weight, conv, columns *cuda.Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	if kernel < 1 || dilation < 1 {
		return fmt.Errorf("cuda: invalid conv kernel=%d dilation=%d", kernel, dilation)
	}
	if blas == nil {
		return fmt.Errorf("cuda: Conv1dForwardTo requires a cuBLAS handle")
	}
	if err := Im2col(x, columns, batch, channels, length, kernel, dilation); err != nil {
		return err
	}
	columnCount := channels * kernel
	return blas.SgemmRowMajorNT(outChannels, batch*length, columnCount, 1,
		weight.Pointer(), columnCount, columns.Pointer(), columnCount, 0, conv.Pointer(), batch*length)
}

// ConvWeightGradGemm accumulates the convolution weight gradient from the
// columns gathered by the forward pass: dw += dy*columns^T with dy
// [outChannels, batch*length].
func ConvWeightGradGemm(blas *cuda.Blas, dy, columns, dw *cuda.Buffer, batch, channels, length, outChannels, kernel int) error {
	if blas == nil {
		return fmt.Errorf("cuda: ConvWeightGradGemm requires a cuBLAS handle")
	}
	columnCount := channels * kernel
	return blas.SgemmRowMajor(outChannels, columnCount, batch*length, 1,
		dy.Pointer(), batch*length, columns.Pointer(), columnCount, 0, dw.Pointer(), columnCount)
}

// ConvInputGrad accumulates the input gradient from dy [outChannels,
// batch*length] into time-major dx [batch,length,channels].
func ConvInputGrad(blas *cuda.Blas, dy, weight, dx *cuda.Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	if blas == nil {
		return fmt.Errorf("cuda: ConvInputGrad requires a cuBLAS handle")
	}
	columnCount := channels * kernel
	columns, err := cuda.Alloc(columnCount * batch * length * 4)
	if err != nil {
		return err
	}
	defer columns.Free()
	if err := blas.SgemmRowMajorTransposeA(batch*length, columnCount, outChannels, 1,
		dy.Pointer(), batch*length, weight.Pointer(), columnCount, 0, columns.Pointer(), columnCount); err != nil {
		return err
	}
	return Col2im(columns, dx, batch, channels, length, kernel, dilation)
}
