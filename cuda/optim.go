package cuda

import (
	"sync"
	"unsafe"
)

const optimKernelSource = `
extern "C" __global__ void sum_squares(const float* x, float* acc, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) atomicAdd(acc, x[i] * x[i]);
}
extern "C" __global__ void sqrt_scalar(float* x) {
	x[0] = sqrtf(x[0]);
}
extern "C" __global__ void adamw_update(float* p, const float* g, float* m, float* v, const float* norm, int n,
	float beta1, float beta2, float stepSize, float biasCorrection2, float weightDecay, float eps, float maxNorm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float coefficient = maxNorm / (norm[0] + 1e-6f);
	if (coefficient > 1.0f) coefficient = 1.0f;
	float grad = g[i] * coefficient;
	m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
	v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
	float denom = sqrtf(v[i] / biasCorrection2) + eps;
	p[i] = p[i] * (1.0f - weightDecay) - stepSize * m[i] / denom;
}
`

type optimKernelSet struct {
	sumSquares *Kernel
	sqrtScalar *Kernel
	adamw      *Kernel
}

var (
	optimProgramOnce sync.Once
	optimProgram     *Program
	optimSet         *optimKernelSet
	optimErr         error
)

func optimKernels() (*optimKernelSet, error) {
	optimProgramOnce.Do(func() {
		program, err := Compile(optimKernelSource)
		if err != nil {
			optimErr = err
			return
		}
		set := &optimKernelSet{}
		for name, target := range map[string]**Kernel{
			"sum_squares":  &set.sumSquares,
			"sqrt_scalar":  &set.sqrtScalar,
			"adamw_update": &set.adamw,
		} {
			kernel, err := program.Function(name)
			if err != nil {
				optimErr = err
				return
			}
			*target = kernel
		}
		optimProgram, optimSet = program, set
	})
	return optimSet, optimErr
}

// SumSquares accumulates the sum of x[i]^2 into acc[0].
func SumSquares(x, acc *Buffer, count int) error {
	set, err := optimKernels()
	if err != nil {
		return err
	}
	xAddr, accAddr := x.Pointer(), acc.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&accAddr), unsafe.Pointer(&n)}
	return set.sumSquares.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// SqrtScalar replaces x[0] with its square root.
func SqrtScalar(x *Buffer) error {
	set, err := optimKernels()
	if err != nil {
		return err
	}
	xAddr := x.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&xAddr)}
	return set.sqrtScalar.Launch([3]int{1, 1, 1}, [3]int{1, 1, 1}, 0, nil, args)
}

// AdamWUpdate applies one AdamW step to p from g, updating the moment buffers
// m and v and clipping the gradient by the global norm in norm[0].
func AdamWUpdate(p, g, m, v, norm *Buffer, count int, beta1, beta2, stepSize, biasCorrection2, weightDecay, eps, maxNorm float32) error {
	set, err := optimKernels()
	if err != nil {
		return err
	}
	pAddr, gAddr, mAddr, vAddr, normAddr := p.Pointer(), g.Pointer(), m.Pointer(), v.Pointer(), norm.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{
		unsafe.Pointer(&pAddr), unsafe.Pointer(&gAddr), unsafe.Pointer(&mAddr), unsafe.Pointer(&vAddr), unsafe.Pointer(&normAddr),
		unsafe.Pointer(&n), unsafe.Pointer(&beta1), unsafe.Pointer(&beta2), unsafe.Pointer(&stepSize),
		unsafe.Pointer(&biasCorrection2), unsafe.Pointer(&weightDecay), unsafe.Pointer(&eps), unsafe.Pointer(&maxNorm),
	}
	return set.adamw.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}
