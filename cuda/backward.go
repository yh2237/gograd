package cuda

import (
	"sync"
	"unsafe"
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
extern "C" __global__ void conv_bias_grad(const float* dy, float* db, int batch, int outChannels, int length) {
	// One thread per (batch, out) scans the time axis, then combines across
	// batches with an atomic.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= batch * outChannels) return;
	int o = index % outChannels;
	int b = index / outChannels;
	float sum = 0.0f;
	for (int t = 0; t < length; t++) sum += dy[(b * outChannels + o) * length + t];
	if (sum != 0.0f) atomicAdd(&db[o], sum);
}
extern "C" __global__ void conv_weight_grad(const float* x, const float* dy, float* dw, int batch, int channels, int length, int outChannels, int kernel, int dilation) {
	// One thread per (batch, out, in, kernel) keeps the grid large and lets each
	// thread scan only the time axis before an atomic add.
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
	if (sum != 0.0f) atomicAdd(&dw[ock], sum);
}
extern "C" __global__ void conv_input_grad(const float* dy, const float* w, float* dx, int batch, int channels, int length, int outChannels, int kernel, int dilation) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch * channels * length;
	if (index >= total) return;
	int t = index % length;
	int c = (index / length) % channels;
	int b = index / (channels * length);
	float sum = 0.0f;
	for (int o = 0; o < outChannels; o++)
		for (int k = 0; k < kernel; k++) {
			int dst = t + dilation - k * dilation;
			if (dst < 0 || dst >= length) continue;
			sum += dy[(b * outChannels + o) * length + dst] * w[(o * channels + c) * kernel + k];
		}
	dx[index] += sum;
}
`

type backwardKernelSet struct {
	tanhBackward   *Kernel
	addInto        *Kernel
	addPair        *Kernel
	columnSum      *Kernel
	convBiasGrad   *Kernel
	convWeightGrad *Kernel
	convInputGrad  *Kernel
}

var (
	backwardProgramOnce sync.Once
	backwardProgram     *Program
	backwardSet         *backwardKernelSet
	backwardErr         error
)

func backwardKernels() (*backwardKernelSet, error) {
	backwardProgramOnce.Do(func() {
		program, err := Compile(backwardKernelSource)
		if err != nil {
			backwardErr = err
			return
		}
		set := &backwardKernelSet{}
		for name, target := range map[string]**Kernel{
			"tanh_backward":    &set.tanhBackward,
			"add_into":         &set.addInto,
			"add_pair":         &set.addPair,
			"column_sum":       &set.columnSum,
			"conv_bias_grad":   &set.convBiasGrad,
			"conv_weight_grad": &set.convWeightGrad,
			"conv_input_grad":  &set.convInputGrad,
		} {
			kernel, err := program.Function(name)
			if err != nil {
				backwardErr = err
				return
			}
			*target = kernel
		}
		backwardProgram, backwardSet = program, set
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
func TanhBackward(a, dout, din *Buffer, count int) error {
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
func AddInto(dst, src *Buffer, count int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	dstAddr, srcAddr := dst.Pointer(), src.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{unsafe.Pointer(&dstAddr), unsafe.Pointer(&srcAddr), unsafe.Pointer(&n)}
	return set.addInto.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// AddPair writes dst = a + b elementwise. Unlike AddInto it needs no zeroed
// destination, so it replaces a copy plus an accumulate.
func AddPair(dst, a, b *Buffer, count int) error {
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
func ColumnSum(x, out *Buffer, rows, cols int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	chunks := 1
	if rows > 1 {
		chunks = 64
		if chunks > rows {
			chunks = rows
		}
	}
	xAddr, outAddr := x.Pointer(), out.Pointer()
	args := int32Args(int32(rows), int32(cols), int32(chunks))
	args = append([]unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&outAddr)}, args...)
	return set.columnSum.Launch(elementGrid(cols*chunks), [3]int{256, 1, 1}, 0, nil, args)
}

// ConvBiasGrad accumulates sum of dy into db[outChannels].
func ConvBiasGrad(dy, db *Buffer, batch, outChannels, length int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	dyAddr, dbAddr := dy.Pointer(), db.Pointer()
	args := int32Args(int32(batch), int32(outChannels), int32(length))
	args = append([]unsafe.Pointer{unsafe.Pointer(&dyAddr), unsafe.Pointer(&dbAddr)}, args...)
	return set.convBiasGrad.Launch(elementGrid(batch*outChannels), [3]int{256, 1, 1}, 0, nil, args)
}

// ConvWeightGrad accumulates the convolution weight gradient.
func ConvWeightGrad(x, dy, dw *Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	xAddr, dyAddr, dwAddr := x.Pointer(), dy.Pointer(), dw.Pointer()
	args := int32Args(int32(batch), int32(channels), int32(length), int32(outChannels), int32(kernel), int32(dilation))
	args = append([]unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&dyAddr), unsafe.Pointer(&dwAddr)}, args...)
	return set.convWeightGrad.Launch(elementGrid(batch*outChannels*channels*kernel), [3]int{256, 1, 1}, 0, nil, args)
}

// ConvInputGrad accumulates the convolution gradient with respect to its input.
func ConvInputGrad(dy, weight, dx *Buffer, batch, channels, length, outChannels, kernel, dilation int) error {
	set, err := backwardKernels()
	if err != nil {
		return err
	}
	dyAddr, wAddr, dxAddr := dy.Pointer(), weight.Pointer(), dx.Pointer()
	args := int32Args(int32(batch), int32(channels), int32(length), int32(outChannels), int32(kernel), int32(dilation))
	args = append([]unsafe.Pointer{unsafe.Pointer(&dyAddr), unsafe.Pointer(&wAddr), unsafe.Pointer(&dxAddr)}, args...)
	return set.convInputGrad.Launch(elementGrid(batch*channels*length), [3]int{256, 1, 1}, 0, nil, args)
}
