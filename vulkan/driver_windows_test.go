//go:build windows && amd64

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
