package kernels

import (
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

const backwardKernelSource = `
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
`

type backwardKernelSet struct {
	columnSum *cuda.Kernel
}

var backwardCache cuda.DeviceCache[*backwardKernelSet]

func backwardKernels() (*backwardKernelSet, error) {
	return backwardCache.Get(func() (*backwardKernelSet, error) {
		program, err := cuda.Compile(backwardKernelSource)
		if err != nil {
			return nil, err
		}
		kernel, err := program.Function("column_sum")
		if err != nil {
			_ = program.Close()
			return nil, err
		}
		return &backwardKernelSet{columnSum: kernel}, nil
	})
}

func int32Args(values ...int32) []unsafe.Pointer {
	args := make([]unsafe.Pointer, len(values))
	for i := range values {
		args[i] = unsafe.Pointer(&values[i])
	}
	return args
}

// ColumnSum accumulates out[c] += sum_r x[r,c] for a [rows,cols] matrix. The
// caller zeroes out before the first accumulation.
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
