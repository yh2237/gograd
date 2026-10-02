package kernels

import (
	"github.com/yh2237/gograd/cuda"
	"sync"
	"unsafe"
)

const opsKernelSource = `
extern "C" __global__ void activation(const float* x, const float* grad, float* out, int n, int kind, float slope, int backward) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float v = x[i], y;
	if (kind == 0) y = backward ? (v > 0.0f ? 1.0f : 0.0f) : fmaxf(v, 0.0f);
	else if (kind == 1) y = backward ? (v > 0.0f ? 1.0f : slope) : (v > 0.0f ? v : slope * v);
	else {
		float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
		float th = tanhf(u);
		y = backward ? 0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v) : 0.5f * v * (1.0f + th);
	}
	out[i] = backward ? grad[i] * y : y;
}
extern "C" __global__ void add_tanh(const float* a, const float* b, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = tanhf(a[i] + b[i]);
}
extern "C" __global__ void tanh_array(const float* x, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = tanhf(x[i]);
}
extern "C" __global__ void bias_columns_tanh(float* y, const float* bias, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows * cols) return;
	y[i] = tanhf(y[i] + bias[i % cols]);
}
extern "C" __global__ void add_bias_columns(float* y, const float* bias, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows * cols) return;
	y[i] += bias[i % cols];
}
extern "C" __global__ void transpose12(const float* x, float* out, int d0, int d1, int d2) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= d0 * d1 * d2) return;
	int k = index % d1;
	int rest = index / d1;
	int j = rest % d2;
	int b = rest / d2;
	out[index] = x[b * d1 * d2 + k * d2 + j];
}
`

type opsKernelSet struct {
	addTanh    *cuda.Kernel
	biasTanh   *cuda.Kernel
	biasColumn *cuda.Kernel
	transpose  *cuda.Kernel
	tanh       *cuda.Kernel
	activation *cuda.Kernel
}

var (
	opsProgramOnce sync.Once
	opsProgram     *cuda.Program
	opsSet         *opsKernelSet
	opsProgramErr  error
)

func opsKernels() (*opsKernelSet, error) {
	opsProgramOnce.Do(func() {
		program, err := cuda.Compile(opsKernelSource)
		if err != nil {
			opsProgramErr = err
			return
		}
		set := &opsKernelSet{}
		for name, target := range map[string]**cuda.Kernel{
			"add_tanh":          &set.addTanh,
			"bias_columns_tanh": &set.biasTanh,
			"add_bias_columns":  &set.biasColumn,
			"transpose12":       &set.transpose,
			"tanh_array":        &set.tanh,
			"activation":        &set.activation,
		} {
			kernel, err := program.Function(name)
			if err != nil {
				opsProgramErr = err
				return
			}
			*target = kernel
		}
		opsProgram, opsSet = program, set
	})
	return opsSet, opsProgramErr
}

// Activation applies ReLU (0), LeakyReLU (1), or GELU (2), or their derivatives.
func Activation(x, grad, out *cuda.Buffer, count, kind int, slope float32, backward bool) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	xAddr, outAddr := x.Pointer(), out.Pointer()
	var gradAddr uintptr
	if grad != nil {
		gradAddr = grad.Pointer()
	}
	n, k, b := int32(count), int32(kind), int32(0)
	if backward {
		b = 1
	}
	args := []unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&gradAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&n), unsafe.Pointer(&k), unsafe.Pointer(&slope), unsafe.Pointer(&b)}
	return set.activation.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

func elementGrid(count int) [3]int {
	return [3]int{(count + 255) / 256, 1, 1}
}

// AddTanh writes tanh(a+b) elementwise for equal-length buffers.
func AddTanh(a, b, out *cuda.Buffer, count int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	n := int32(count)
	aAddr, bAddr, outAddr := a.Pointer(), b.Pointer(), out.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&aAddr), unsafe.Pointer(&bAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&n)}
	return set.addTanh.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// Tanh writes tanh(x) elementwise.
func Tanh(x, out *cuda.Buffer, count int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	xAddr, outAddr := x.Pointer(), out.Pointer()
	n := int32(count)
	args := []unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&n)}
	return set.tanh.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// BiasColumnsTanh adds a per-column bias and applies tanh to a [rows,cols]
// row-major matrix.
func BiasColumnsTanh(y, bias *cuda.Buffer, rows, cols int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	r, c := int32(rows), int32(cols)
	yAddr, biasAddr := y.Pointer(), bias.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&yAddr), unsafe.Pointer(&biasAddr), unsafe.Pointer(&r), unsafe.Pointer(&c)}
	return set.biasTanh.Launch(elementGrid(rows*cols), [3]int{256, 1, 1}, 0, nil, args)
}

// AddBiasColumns adds a per-column bias to a [rows,cols] row-major matrix.
func AddBiasColumns(y, bias *cuda.Buffer, rows, cols int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	r, c := int32(rows), int32(cols)
	yAddr, biasAddr := y.Pointer(), bias.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&yAddr), unsafe.Pointer(&biasAddr), unsafe.Pointer(&r), unsafe.Pointer(&c)}
	return set.biasColumn.Launch(elementGrid(rows*cols), [3]int{256, 1, 1}, 0, nil, args)
}

// Transpose12 swaps the last two dimensions of a contiguous [d0,d1,d2] tensor.
func Transpose12(x, out *cuda.Buffer, d0, d1, d2 int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	a, b, c := int32(d0), int32(d1), int32(d2)
	xAddr, outAddr := x.Pointer(), out.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&c)}
	return set.transpose.Launch(elementGrid(d0*d1*d2), [3]int{256, 1, 1}, 0, nil, args)
}
