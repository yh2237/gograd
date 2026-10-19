package vulkan

import (
	"math/rand/v2"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

// BenchmarkVulkanBinaryViews compares the strided view operation with the
// fallback that materializes both operands first.
func BenchmarkVulkanBinaryViews(b *testing.B) {
	backend := NewBackend()
	if !backend.Available() {
		backend.Close()
		b.Skip("vulkan unavailable")
	}
	defer backend.Close()
	data := make([]float32, 74*768)
	for i := range data {
		data[i] = rand.New(rand.NewPCG(uint64(i), 7)).Float32() - .5
	}
	left, err := backend.Upload(data)
	if err != nil {
		b.Fatal(err)
	}
	defer left.Close()
	right, err := backend.Upload(data[:768])
	if err != nil {
		b.Fatal(err)
	}
	defer right.Close()
	shape := []int{74, 768}
	out, err := backend.Alloc(74 * 768)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()
	leftView := gpu.View{Buffer: left, Shape: shape, Strides: []int{768, 1}}
	rightView := gpu.View{Buffer: right, Shape: shape, Strides: []int{0, 1}}
	b.Run("views", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := backend.AddViews(out, leftView, rightView); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("materialized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			expanded, err := backend.Alloc(74 * 768)
			if err != nil {
				b.Fatal(err)
			}
			if err := backend.Gather(expanded, rightView); err != nil {
				b.Fatal(err)
			}
			if err := backend.Add(out, left, expanded, 74*768); err != nil {
				b.Fatal(err)
			}
			expanded.Close()
		}
	})
}
