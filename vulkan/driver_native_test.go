//go:build (windows || (linux && !android)) && amd64

package vulkan

import (
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	"testing"
)

func TestVulkanABI(t *testing.T) {
	if err := checkABI(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanTwoDimensionalDispatch(t *testing.T) {
	b := testBackend(t)
	// Force a valid smaller x limit to exercise y indexing without allocating
	// a huge tensor. The actual GPU supports at least this many groups.
	b.driver.(*vkDriver).properties.MaxWorkGroupCount[0] = 2
	values, want := make([]float32, 513), make([]float32, 513)
	for i := range values {
		values[i] = float32(i) / 8
		want[i] = values[i] * .5
	}
	x, out := testUpload(t, b, values), testAlloc(t, b, len(values))
	if err := b.MulScalar(out, x, len(values), .5); err != nil {
		t.Fatal(err)
	}
	checkValues(t, b, out, want)
}

func TestVulkanTwoDimensionalGEMM(t *testing.T) {
	b := testBackend(t)
	b.driver.(*vkDriver).properties.MaxWorkGroupCount[0] = 5
	a, right, want := make([]float32, 3*9*13), make([]float32, 13*17), make([]float32, 3*9*17)
	for i := range a {
		a[i] = float32(i%17-8) / 16
	}
	for i := range right {
		right[i] = float32(i%13-6) / 8
	}
	tensor.SGEMM(want, a, right, 3*9, 17, 13)
	av := gpu.View{Buffer: testUpload(t, b, a), Shape: []int{3, 9, 13}, Strides: []int{117, 13, 1}}
	bv := gpu.View{Buffer: testUpload(t, b, right), Shape: []int{13, 17}, Strides: []int{17, 1}}
	out := testAlloc(t, b, len(want))
	if err := b.MatMulViews(out, av, bv); err != nil {
		t.Fatal(err)
	}
	checkMatrix(t, b, out, want)
}

func TestVulkanTwoDimensionalTransformer(t *testing.T) {
	b := testBackend(t)
	b.driver.(*vkDriver).properties.MaxWorkGroupCount[0] = 2
	data, want := make([]float32, 7*13), make([]float32, 7*13)
	for i := range data {
		data[i] = float32(i / 13)
	}
	for i := range want {
		want[i] = 1.0 / 13
	}
	out := testAlloc(t, b, len(data))
	if err := b.Softmax(out, testUpload(t, b, data), 7, 13); err != nil {
		t.Fatal(err)
	}
	checkMatrix(t, b, out, want)
	x, w := make([]float32, 333*3), make([]float32, 5*3*3)
	for i := range x {
		x[i] = 1
	}
	for i := range w {
		w[i] = 1
	}
	want = make([]float32, 333*5)
	for i := range want {
		want[i] = 9
		if i/5 == 0 || i/5 == 332 {
			want[i] = 6
		}
	}
	out = testAlloc(t, b, len(want))
	if err := b.Conv1d(out, testUpload(t, b, x), testUpload(t, b, w), nil, gpu.Conv1dOptions{Batch: 1, InputLength: 333, InChannels: 3, OutChannels: 5, Kernel: 3, Stride: 1, Padding: 1, Dilation: 1}); err != nil {
		t.Fatal(err)
	}
	checkMatrix(t, b, out, want)
}
