package autograd

import "github.com/yh2237/gograd/tensor"

// ElementwiseKernel pairs the CPU scalar rule and CUDA kernel opcode. The
// registry is the dispatch point for elementwise forward and backward rules.
type ElementwiseKernel struct {
	CPUForward, CPUGradA, CPUGradB func(float32, float32) float32
	CUDAOpcode                     int
}

var elementwiseRegistry = map[string]ElementwiseKernel{
	"add": {func(x, y float32) float32 { return x + y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return 1 }, 0},
	"sub": {func(x, y float32) float32 { return x - y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return -1 }, 1},
	"mul": {func(x, y float32) float32 { return x * y }, func(x, y float32) float32 { return y }, func(x, y float32) float32 { return x }, 2},
	"div": {func(x, y float32) float32 { return x / y }, func(x, y float32) float32 { return 1 / y }, func(x, y float32) float32 { return -x / (y * y) }, 3},
}

func dispatchElementwise(name string, a, b *Tensor) *Tensor {
	spec, ok := elementwiseRegistry[name]
	if !ok {
		panic("autograd: unregistered operation " + name)
	}
	same(a, b)
	a, b = a.Contiguous(), b.Contiguous()
	if a.Device == tensor.CUDA {
		return gpuBinary(a, b, spec.CUDAOpcode)
	}
	return binary(a, b, spec.CPUForward, spec.CPUGradA, spec.CPUGradB)
}

// RegisteredElementwiseOps lists operations handled by this dispatch table.
func RegisteredElementwiseOps() []string { return []string{"add", "sub", "mul", "div"} }
