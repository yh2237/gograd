package cuda

import (
	"sync"
	"unsafe"
)

const opsKernelSource = `
extern "C" __global__ void add_tanh(const float* a, const float* b, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = tanhf(a[i] + b[i]);
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
	addTanh    *Kernel
	biasTanh   *Kernel
	biasColumn *Kernel
	transpose  *Kernel
}

var (
	opsProgramOnce sync.Once
	opsProgram     *Program
	opsSet         *opsKernelSet
	opsProgramErr  error
)

func opsKernels() (*opsKernelSet, error) {
	opsProgramOnce.Do(func() {
		program, err := Compile(opsKernelSource)
		if err != nil {
			opsProgramErr = err
			return
		}
		set := &opsKernelSet{}
		for name, target := range map[string]**Kernel{
			"add_tanh":          &set.addTanh,
			"bias_columns_tanh": &set.biasTanh,
			"add_bias_columns":  &set.biasColumn,
			"transpose12":       &set.transpose,
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

func elementGrid(count int) [3]int {
	return [3]int{(count + 255) / 256, 1, 1}
}

// AddTanh writes tanh(a+b) elementwise for equal-length buffers.
func AddTanh(a, b, out *Buffer, count int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	n := int32(count)
	aAddr, bAddr, outAddr := a.Pointer(), b.Pointer(), out.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&aAddr), unsafe.Pointer(&bAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&n)}
	return set.addTanh.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}

// BiasColumnsTanh adds a per-column bias and applies tanh to a [rows,cols]
// row-major matrix.
func BiasColumnsTanh(y, bias *Buffer, rows, cols int) error {
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
func AddBiasColumns(y, bias *Buffer, rows, cols int) error {
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
func Transpose12(x, out *Buffer, d0, d1, d2 int) error {
	set, err := opsKernels()
	if err != nil {
		return err
	}
	a, b, c := int32(d0), int32(d1), int32(d2)
	xAddr, outAddr := x.Pointer(), out.Pointer()
	args := []unsafe.Pointer{unsafe.Pointer(&xAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&c)}
	return set.transpose.Launch(elementGrid(d0*d1*d2), [3]int{256, 1, 1}, 0, nil, args)
}
