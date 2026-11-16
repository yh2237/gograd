package kernels

import (
	"fmt"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
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
extern "C" __global__ void tanh_array(const float* x, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = tanhf(x[i]);
}
extern "C" __global__ void add_bias_columns(float* y, const float* bias, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows * cols) return;
	y[i] += bias[i % cols];
}
`

type opsKernelSet struct {
	biasColumn *cuda.Kernel
	tanh       *cuda.Kernel
	activation *cuda.Kernel
}

var opsCache cuda.DeviceCache[*opsKernelSet]

func opsKernels() (*opsKernelSet, error) {
	return opsCache.Get(func() (*opsKernelSet, error) {
		program, err := cuda.Compile(opsKernelSource)
		if err != nil {
			return nil, err
		}
		set := &opsKernelSet{}
		for name, target := range map[string]**cuda.Kernel{
			"add_bias_columns": &set.biasColumn,
			"tanh_array":       &set.tanh,
			"activation":       &set.activation,
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

// Activation applies ReLU (0), LeakyReLU (1), or GELU (2), or their derivatives.
func Activation(x, grad, out *cuda.Buffer, count, kind int, slope float32, backward bool) error {
	if backward && grad == nil {
		return fmt.Errorf("kernels: activation backward requires a gradient buffer")
	}
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

// elementGrid maps an element count to a launch grid. cuLaunchKernel rejects
// a zero-sized grid, so an empty tensor still gets one block; the kernels
// guard every element against the count.
func elementGrid(count int) [3]int {
	if count <= 0 {
		return [3]int{1, 1, 1}
	}
	return [3]int{(count + 255) / 256, 1, 1}
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
