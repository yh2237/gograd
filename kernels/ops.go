package kernels

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

const opsKernelSource = `
extern "C" __global__ void add_bias_columns(float* y, const float* bias, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows * cols) return;
	y[i] += bias[i % cols];
}
`

type opsKernelSet struct {
	biasColumn *cuda.Kernel
}

var opsCache cuda.DeviceCache[*opsKernelSet]

func opsKernels() (*opsKernelSet, error) {
	return opsCache.Get(func() (*opsKernelSet, error) {
		program, err := cuda.Compile(opsKernelSource)
		if err != nil {
			return nil, err
		}
		kernel, err := program.Function("add_bias_columns")
		if err != nil {
			_ = program.Close()
			return nil, err
		}
		return &opsKernelSet{biasColumn: kernel}, nil
	})
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
