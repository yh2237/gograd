package irodori

import (
	"fmt"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
)

// LinearRowsCUDA runs one dense projection on gograd's CUDA matmul kernel.
// It uploads only this layer's weights, keeping peak device memory small.
// The caller must hold a CUDAContext on the current OS thread.
func LinearRowsCUDA(x, w []float32, rows, in, out int) ([]float32, error) {
	if rows <= 0 || in <= 0 || out <= 0 || len(x) != rows*in || len(w) != out*in {
		return nil, fmt.Errorf("irodori: invalid CUDA projection shape")
	}
	transposed := make([]float32, len(w))
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			transposed[i*out+o] = w[o*in+i]
		}
	}
	a, err := autograd.New(x, []int{rows, in}, tensor.CUDA, false)
	if err != nil {
		return nil, err
	}
	defer a.Close()
	b, err := autograd.New(transposed, []int{in, out}, tensor.CUDA, false)
	if err != nil {
		return nil, err
	}
	defer b.Close()
	var result *autograd.Tensor
	autograd.NoGrad(func() { result = autograd.MatMul(a, b) })
	defer result.Close()
	return result.ToHost()
}
