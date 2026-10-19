//go:build (windows || (linux && !android)) && amd64

package vulkan

import "testing"

func TestVulkanFusedBiasTwoDimensionalDispatch(t *testing.T) {
	b := testBackend(t)
	b.mu.Lock()
	b.driver.(*vkDriver).properties.MaxWorkGroupCount[0] = 2
	b.mu.Unlock()
	const rows, cols = 5, 129
	x, bias, r := make([]float32, rows*cols), make([]float32, cols), make([]float32, rows*cols)
	want := make([]float32, len(x))
	for i := range x {
		x[i] = float32(i % 23)
		r[i] = float32(i % 11)
	}
	for i := range bias {
		bias[i] = float32(i % 7)
	}
	for i := range want {
		want[i] = x[i] + bias[i%cols] + r[i]
	}
	y := testAlloc(t, b, len(x))
	if err := b.BiasResidual(y, testUpload(t, b, x), testUpload(t, b, bias), testUpload(t, b, r), rows, cols); err != nil {
		t.Fatal(err)
	}
	checkValues(t, b, y, want)
}
