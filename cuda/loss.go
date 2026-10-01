package cuda

import (
	"runtime"
	"sync"
	"unsafe"
)

// sequenceLossSource computes the masked Smooth L1 loss with per-row median
// centering and the adjacent-frame delta term, and writes d loss / d predicted.
// One block handles one row. The batch-wide valid and pair counts are passed in
// so the reductions match a single batch-level mean. Shared layout: sp[time],
// st[time], ga[time], gd[time], red[blockDim], then a small scalar area.
const sequenceLossSource = `
__device__ float block_sum(float v, float* red) {
	int tid = threadIdx.x;
	red[tid] = v;
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) red[tid] += red[tid + s];
		__syncthreads();
	}
	float result = red[0];
	__syncthreads();
	return result;
}

extern "C" __global__ void sequence_loss_grad(
	const float* predicted, const float* target, const unsigned char* mask,
	float* gradient, float* lossOut,
	int rows, int time, int totalValid, int totalPairs,
	int bounded, float low, float high, float deltaWeight) {
	extern __shared__ float smem[];
	float* sp = smem;
	float* st = smem + time;
	float* ga = smem + 2 * time;
	float* gd = smem + 3 * time;
	float* red = smem + 4 * time;
	float* sc = red + blockDim.x;
	int* flags = (int*)(sc + 4);

	int row = blockIdx.x;
	int tid = threadIdx.x;
	const float* rowPredicted = predicted + row * time;
	const float* rowTarget = target + row * time;
	const unsigned char* rowMask = mask + row * time;

	float masked = 0.0f;
	for (int t = tid; t < time; t += blockDim.x) {
		sp[t] = rowPredicted[t];
		st[t] = rowTarget[t];
		if (rowMask[t]) masked += 1.0f;
	}
	if (tid == 0) {
		sc[0] = 0.0f;
		sc[1] = 0.0f;
		flags[0] = 0;
		flags[1] = 0;
	}
	__syncthreads();
	float count = block_sum(masked, red);
	int n = (int)count;
	int k = (n - 1) / 2;

	for (int t = tid; t < time; t += blockDim.x) {
		if (!rowMask[t] || n == 0) continue;
		float v = sp[t];
		int less = 0, equal = 0;
		for (int u = 0; u < time; u++) {
			if (!rowMask[u]) continue;
			float other = sp[u];
			if (other < v) less++;
			else if (other == v) equal++;
		}
		if (less <= k && k < less + equal) {
			int old = atomicExch(&flags[0], 1);
			if (old == 0) sc[0] = v;
		}
	}
	__syncthreads();
	float medianP = sc[0];

	for (int t = tid; t < time; t += blockDim.x) {
		if (!rowMask[t] || n == 0) continue;
		float v = st[t];
		int less = 0, equal = 0;
		for (int u = 0; u < time; u++) {
			if (!rowMask[u]) continue;
			float other = st[u];
			if (other < v) less++;
			else if (other == v) equal++;
		}
		if (less <= k && k < less + equal) {
			int old = atomicExch(&flags[1], 1);
			if (old == 0) sc[1] = v;
		}
	}
	__syncthreads();
	float medianT = sc[1];

	for (int t = tid; t < time; t += blockDim.x) {
		sp[t] -= medianP;
		float value = st[t] - medianT;
		if (bounded) {
			if (value < low) value = low;
			if (value > high) value = high;
		}
		st[t] = value;
	}
	float equalFlag = 0.0f;
	for (int t = tid; t < time; t += blockDim.x) {
		if (rowMask[t] && n > 0 && sp[t] == 0.0f) equalFlag += 1.0f;
	}
	__syncthreads();
	float equalCount = block_sum(equalFlag, red);

	float absoluteLoss = 0.0f;
	for (int t = tid; t < time; t += blockDim.x) {
		float g = 0.0f;
		if (rowMask[t]) {
			float d = sp[t] - st[t];
			float ad = fabsf(d);
			if (ad < 1.0f) {
				absoluteLoss += 0.5f * d * d;
				g = d;
			} else {
				absoluteLoss += ad - 0.5f;
				g = d < 0.0f ? -1.0f : 1.0f;
			}
		}
		ga[t] = g;
		gd[t] = 0.0f;
	}
	__syncthreads();
	float absoluteSum = block_sum(absoluteLoss, red);

	float deltaLoss = 0.0f;
	for (int t = tid; t < time; t += blockDim.x) {
		if (t < 1) continue;
		if (!(rowMask[t] && rowMask[t - 1])) continue;
		float d = (sp[t] - sp[t - 1]) - (st[t] - st[t - 1]);
		float ad = fabsf(d);
		float g;
		if (ad < 1.0f) {
			deltaLoss += 0.5f * d * d;
			g = d;
		} else {
			deltaLoss += ad - 0.5f;
			g = d < 0.0f ? -1.0f : 1.0f;
		}
		atomicAdd(&gd[t], g);
		atomicAdd(&gd[t - 1], -g);
	}
	__syncthreads();
	float deltaSum = block_sum(deltaLoss, red);

	float gradSum = 0.0f;
	for (int t = tid; t < time; t += blockDim.x) {
		float value = 0.0f;
		if (totalValid > 0) value = ga[t] / (float)totalValid;
		if (totalPairs > 0) value += deltaWeight * gd[t] / (float)totalPairs;
		ga[t] = value;
		gradSum += value;
	}
	__syncthreads();
	float rowGradSum = block_sum(gradSum, red);

	for (int t = tid; t < time; t += blockDim.x) {
		float value = ga[t];
		if (rowMask[t] && totalValid > 0 && sp[t] == 0.0f && equalCount > 0.0f) {
			value -= rowGradSum / equalCount;
		}
		gradient[row * time + t] = value;
	}
	if (tid == 0) {
		float absolute = totalValid > 0 ? absoluteSum / (float)totalValid : 0.0f;
		float delta = totalPairs > 0 ? deltaSum / (float)totalPairs : 0.0f;
		atomicAdd(lossOut, absolute + deltaWeight * delta);
	}
}
`

const sequenceLossBlock = 256

var (
	lossProgramOnce sync.Once
	lossProgram     *Program
	lossKernel      *Kernel
	lossProgramErr  error
)

func lossKernels() (*Kernel, error) {
	lossProgramOnce.Do(func() {
		program, err := Compile(sequenceLossSource)
		if err != nil {
			lossProgramErr = err
			return
		}
		kernel, err := program.Function("sequence_loss_grad")
		if err != nil {
			lossProgramErr = err
			return
		}
		lossProgram, lossKernel = program, kernel
	})
	return lossKernel, lossProgramErr
}

// SequenceLossGrad computes the masked loss and its gradient with respect to
// predicted. mask holds rows*time bytes (1 for voiced frames); totalValid and
// totalPairs are the batch-wide masked-element and adjacent-pair counts the
// reference loss divides by. loss is a one-element buffer and gradient has
// rows*time floats.
func SequenceLossGrad(predicted, target, mask, gradient, loss *Buffer, rows, time, totalValid, totalPairs int, bounded bool, low, high, deltaWeight float64) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	kernel, err := lossKernels()
	if err != nil {
		return err
	}
	boundedValue := int32(0)
	if bounded {
		boundedValue = 1
	}
	predictedAddr, targetAddr := predicted.Pointer(), target.Pointer()
	maskAddr, gradientAddr, lossAddr := mask.Pointer(), gradient.Pointer(), loss.Pointer()
	r, t := int32(rows), int32(time)
	valid, pairs := int32(totalValid), int32(totalPairs)
	b, l, h, w := boundedValue, float32(low), float32(high), float32(deltaWeight)
	args := []unsafe.Pointer{
		unsafe.Pointer(&predictedAddr), unsafe.Pointer(&targetAddr), unsafe.Pointer(&maskAddr),
		unsafe.Pointer(&gradientAddr), unsafe.Pointer(&lossAddr),
		unsafe.Pointer(&r), unsafe.Pointer(&t), unsafe.Pointer(&valid), unsafe.Pointer(&pairs),
		unsafe.Pointer(&b), unsafe.Pointer(&l), unsafe.Pointer(&h), unsafe.Pointer(&w),
	}
	shared := (4*time + sequenceLossBlock + 8) * 4
	return kernel.Launch([3]int{rows, 1, 1}, [3]int{sequenceLossBlock, 1, 1}, shared, nil, args)
}
