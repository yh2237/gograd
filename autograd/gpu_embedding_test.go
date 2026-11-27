package autograd

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func TestGPUEmbedding(t *testing.T) {
	for _, device := range []tensor.Device{tensor.Vulkan, tensor.DirectML} {
		t.Run(string(device), func(t *testing.T) {
			if !gpu.Available(string(device)) {
				t.Skipf("%s unavailable", device)
			}
			exec := NewExecutionContext()
			weights := make([]float32, 16*12)
			for i := range weights {
				weights[i] = float32(i%37-18) / 4
			}
			cpuW, err := exec.New(weights, []int{16, 12}, tensor.CPU, false)
			if err != nil {
				t.Fatal(err)
			}
			defer cpuW.Close()
			gpuW, err := exec.New(weights, []int{16, 12}, device, true)
			if err != nil {
				t.Fatal(err)
			}
			defer gpuW.Close()

			assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() {
				Embedding(gpuW, []int{1, 3}, []int{2})
			})

			cases := []struct {
				name  string
				ids   []int
				shape []int
			}{
				{"contiguous", []int{2, 3, 4, 5}, []int{2, 2}},
				{"single", []int{7}, []int{1}},
				{"scattered", []int{5, 0, 5, 15, 2, 1}, []int{2, 3}},
				{"empty", []int{}, []int{0}},
			}
			exec.NoGrad(func() {
				for _, tc := range cases {
					t.Run(tc.name, func(t *testing.T) {
						want := EmbeddingWithPadding(cpuW, tc.ids, tc.shape, 0)
						defer want.ReleaseGraph()
						got := EmbeddingWithPadding(gpuW, tc.ids, tc.shape, 0)
						defer got.ReleaseGraph()
						if !slices.Equal(got.Shape, want.Shape) {
							t.Fatalf("shape=%v want %v", got.Shape, want.Shape)
						}
						gotVals, err := got.ToHost()
						if err != nil {
							t.Fatal(err)
						}
						wantVals, err := want.ToHost()
						if err != nil {
							t.Fatal(err)
						}
						if !slices.Equal(gotVals, wantVals) {
							t.Fatalf("mismatch for %s", tc.name)
						}
					})
				}
				// Non-contiguous transposed weight
				twCPU := Transpose(cpuW, 0, 1)
				defer twCPU.ReleaseGraph()
				twGPU := Transpose(gpuW, 0, 1)
				defer twGPU.ReleaseGraph()
				want := Embedding(twCPU, []int{3, 1, 11}, []int{3})
				defer want.ReleaseGraph()
				got := Embedding(twGPU, []int{3, 1, 11}, []int{3})
				defer got.ReleaseGraph()
				gotVals, err := got.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				wantVals, err := want.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(gotVals, wantVals) {
					t.Fatalf("transposed weight embedding mismatch")
				}
			})

			layer, err := NewEmbeddingLayer(8, 6, 0, device, rand.New(rand.NewSource(1)))
			if err != nil {
				t.Fatal(err)
			}
			defer layer.Weight.Close()
			NoGrad(func() {
				y := layer.Forward([]int{4, 0, 7}, []int{1, 3})
				defer y.ReleaseGraph()
				if !slices.Equal(y.Shape, []int{1, 3, 6}) {
					t.Fatalf("layer shape=%v", y.Shape)
				}
				vals, err := y.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				if len(vals) != 18 {
					t.Fatalf("len=%d", len(vals))
				}
			})
		})
	}
}
