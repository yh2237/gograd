package kernels

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

const backwardKernelSource = `
extern "C" __global__ void tanh_backward(const float* a, const float* dout, float* din, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) din[i] += dout[i] * (1.0f - a[i] * a[i]);
}
extern "C" __global__ void add_into(float* dst, const float* src, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) dst[i] += src[i];
}
extern "C" __global__ void add_pair(float* dst, const float* a, const float* b, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) dst[i] = a[i] + b[i];
}
extern "C" __global__ void column_sum(const float* x, float* out, int rows, int cols, int chunks) {
	// Each thread sums a contiguous row range for one column and adds it with
	// an atomic, so wide reductions still fill the grid.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= cols * chunks) return;
	int chunk = index % chunks;
	int c = index / chunks;
	int begin = (rows * chunk) / chunks;
	int end = (rows * (chunk + 1)) / chunks;
	float sum = 0.0f;
	for (int r = begin; r < end; r++) sum += x[r * cols + c];
	if (sum != 0.0f) atomicAdd(&out[c], sum);
}
extern "C" __global__ void conv_weight_grad_partial(const float* x, const float* dy, float* partial, int batch, int channels, int length, int outChannels, int kernel, int dilation) {
	// Standalone fallback used when the forward columns are not available. One
	// thread per (batch, out, in, kernel) writes its own partial sum.
	int per = outChannels * channels * kernel;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= batch * per) return;
	int ock = index % per;
	int b = index / per;
	int k = ock % kernel;
	int c = (ock / kernel) % channels;
	int o = ock / (kernel * channels);
	float sum = 0.0f;
	for (int t = 0; t < length; t++) {
		int src = t - dilation + k * dilation;
		if (src < 0 || src >= length) continue;
		sum += dy[(b * outChannels + o) * length + t] * x[(b * channels + c) * length + src];
	}
	partial[index] = sum;
}
extern "C" __global__ void sum_over_batch(const float* partial, float* out, int batch, int per) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= per) return;
	float sum = 0.0f;
	for (int b = 0; b < batch; b++) sum += partial[b * per + index];
	out[index] += sum;
}
`

type backwardKernelSet struct {
	tanhBackward   *cuda.Kernel
	addInto        *cuda.Kernel
	addPair        *cuda.Kernel
	columnSum      *cuda.Kernel
	convWeightPart *cuda.Kernel
	sumOverBatch   *cuda.Kernel
}

var (
	backwardProgramOnce sync.Once
	backwardSet         *backwardKernelSet
	backwardErr         error
)

func backwardKernels() (*backwardKernelSet, error) {
	backwardProgramOnce.Do(func() {
		program, err := cuda.Compile(backwardKernelSource)
		if err != nil {
			backwardErr = err
			return
		}
		set := &backwardKernelSet{}
		for name, target := range map[string]**cuda.Kernel{
			"tanh_backward":            &set.tanhBackward,
			"add_into":                 &set.addInto,
			"add_pair":                 &set.addPair,
			"column_sum":               &set.columnSum,
			"conv_weight_grad_partial": &set.convWeightPart,
			"sum_over_batch":           &set.sumOverBatch,
		} {
			kernel, err := program.Function(name)
			if err != nil {
				backwardErr = err
				return
			}
			*target = kernel
		}
		backwardSet = set
	})
	return backwardSet, backwardErr
}

func int32Args(values ...int32) []unsafe.Pointer {
	args := make([]unsafe.Pointer, len(values))
	for i := range values {
		args[i] = unsafe.Pointer(&values[i])
	}
	return args
}

// TanhBackward accumulates dout*(1-a^2) into din.
func TanhBackward(a, dout, din *cuda.Buffer, count int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	aAddr, doutAddr, dinAddr := a.Pointer(), dout.Pointer(), din.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{unsafe.Pointer(&aAddr), unsafe.Pointer(&doutAddr), unsafe.Pointer(&dinAddr), unsafe.Pointer(&n)}
	return set.tanhBackward.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// AddInto accumulates src into dst elementwise.
func AddInto(dst, src *cuda.Buffer, count int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	dstAddr, srcAddr := dst.Pointer(), src.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{unsafe.Pointer(&dstAddr), unsafe.Pointer(&srcAddr), unsafe.Pointer(&n)}
	return set.addInto.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// AddPair writes dst = a + b elementwise.
func AddPair(dst, a, b *cuda.Buffer, count int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	dstAddr, aAddr, bAddr := dst.Pointer(), a.Pointer(), b.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{unsafe.Pointer(&dstAddr), unsafe.Pointer(&aAddr), unsafe.Pointer(&bAddr), unsafe.Pointer(&n)}
	return set.addPair.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// ColumnSum accumulates out[c] += sum_r x[r,c] for a [rows,cols] matrix. out
// must be zero before the call.
func ColumnSum(x, out *cuda.Buffer, rows, cols int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	chunks := 1
	if rows > 1 {
		chunks = 256
		if chunks > rows {
			chunks = rows
		}
	}
	xAddr, outAddr := x.Pointer(), out.Pointer()
	args := int32Args(int32(rows), int32(cols), int32(chunks))
	args = append([]unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&outAddr)}, args...)
	return set.columnSum.Launch(elementGrid(cols*chunks), [3]int{256, 1, 1}, 0, nil, args)
}

// ConvWeightGrad accumulates the convolution weight gradient for channel-major
// x [batch,channels,length] and dy [batch,outChannels,length]. A first pass
// writes one partial sum per batch without atomics and a second reduces them.
func ConvWeightGrad(x, dy, dw *cuda.Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	per := outChannels * channels * kernel
	partial, err := cuda.Alloc(batch * per * 4)
	if err != nil {
		return err
	}
	defer partial.Free()

	xAddr, dyAddr, partialAddr := x.Pointer(), dy.Pointer(), partial.Pointer()
	args := int32Args(int32(batch), int32(channels), int32(length), int32(outChannels), int32(kernel), int32(dilation))
	args = append([]unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&dyAddr), unsafe.Pointer(&partialAddr)}, args...)
	if err := set.convWeightPart.Launch(elementGrid(batch*per), [3]int{256, 1, 1}, 0, nil, args); err != nil {
		return err
	}
	partialAddr2, dwAddr := partial.Pointer(), dw.Pointer()
	sumArgs := int32Args(int32(batch), int32(per))
	sumArgs = append([]unsafe.Pointer{unsafe.Pointer(&partialAddr2), unsafe.Pointer(&dwAddr)}, sumArgs...)
	return set.sumOverBatch.Launch(elementGrid(per), [3]int{256, 1, 1}, 0, nil, sumArgs)
}
