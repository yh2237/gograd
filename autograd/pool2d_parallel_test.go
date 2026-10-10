package autograd

import (
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// The output exceeds parallelFor's threshold and partitions cut through a
// plane. A monotonic grid makes the exact winner and overlap counts analytical.
func TestPool2dParallelPlanes(t *testing.T) {
	const planes, h, w = 9, 129, 131
	data := make([]float32, planes*h*w)
	for i := range data {
		data[i] = float32(i)
	}
	x := conv2dTestTensor(t, data, []int{3, 3, h, w}, tensor.CPU, true)
	y := MaxPool2d(x, MaxPool2dOptions{KernelSize: [2]int{2, 2}, Stride: [2]int{1, 1}, Padding: [2]int{1, 1}})
	loss := Sum(y, 0, 1, 2, 3)
	defer loss.ReleaseGraph()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	for plane := 0; plane < planes; plane++ {
		for row := 0; row < h; row++ {
			for col := 0; col < w; col++ {
				want := float32(1)
				if row == h-1 {
					want *= 2
				}
				if col == w-1 {
					want *= 2
				}
				index := (plane*h+row)*w + col
				if x.Grad[index] != want {
					t.Fatalf("parallel gradient[%d] %g want %g", index, x.Grad[index], want)
				}
			}
		}
	}
}
