package vulkan

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func checkMatrix(t *testing.T, b *Backend, out *gpu.Buffer, want []float32) {
	t.Helper()
	got, err := b.Download(out, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want[i])) > 1e-4+1e-4*math.Abs(float64(want[i])) {
			t.Fatalf("[%d]=%g want %g", i, v, want[i])
		}
	}
}

func TestVulkanGEMM(t *testing.T) {
	b := testBackend(t)
	rng := rand.New(rand.NewPCG(73, 19))
	for _, dims := range [][3]int{{1, 1, 1}, {7, 11, 13}, {17, 19, 33}, {0, 5, 3}, {5, 0, 3}, {3, 5, 0}, {768, 768, 74}, {768, 768, 768}} {
		for _, ta := range []bool{false, true} {
			for _, tb := range []bool{false, true} {
				t.Run(fmt.Sprintf("%v/%t/%t", dims, ta, tb), func(t *testing.T) {
					m, n, k := dims[0], dims[1], dims[2]
					a, right, want := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
					for i := range a {
						a[i] = rng.Float32() - .5
					}
					for i := range right {
						right[i] = rng.Float32() - .5
					}
					tensor.SGEMMOp(want, a, right, m, n, k, ta, tb)
					out := testAlloc(t, b, m*n)
					if err := b.MatMul(out, testUpload(t, b, a), testUpload(t, b, right), m, n, k, ta, tb); err != nil {
						t.Fatal(err)
					}
					checkMatrix(t, b, out, want)
				})
			}
		}
	}
	if s := b.MemoryStats(); s.LiveBuffers != 0 {
		t.Fatalf("GEMM leaked: %+v", s)
	}
}

func viewIndex(flat int, shape, strides []int, offset int) int {
	for i := len(shape) - 1; i >= 0; i-- {
		offset += flat % shape[i] * strides[i]
		flat /= shape[i]
	}
	return offset
}

func TestVulkanGatherScatter(t *testing.T) {
	b := testBackend(t)
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	base := testUpload(t, b, data)
	for _, v := range []gpu.View{
		{Buffer: base, Shape: []int{4, 2, 3}, Strides: []int{1, 12, 4}},
		{Buffer: base, Shape: []int{2, 2}, Strides: []int{12, 4}, Offset: 5},
		{Buffer: base, Offset: 3},
		{Buffer: base, Shape: []int{0, 3}, Strides: []int{3, 1}, Offset: 24},
	} {
		n, err := v.Validate(b)
		if err != nil {
			t.Fatal(err)
		}
		out := testAlloc(t, b, n)
		want := make([]float32, n)
		for i := range want {
			want[i] = data[viewIndex(i, v.Shape, v.Strides, v.Offset)]
		}
		if err := b.Gather(out, v); err != nil {
			t.Fatal(err)
		}
		checkValues(t, b, out, want)
		for i := range want {
			want[i] = float32(100 + i)
			data[viewIndex(i, v.Shape, v.Strides, v.Offset)] = want[i]
		}
		if err := b.Scatter(v, testUpload(t, b, want)); err != nil {
			t.Fatal(err)
		}
		checkValues(t, b, base, data)
	}
	expanded := gpu.View{Buffer: base, Shape: []int{2, 3, 5}, Strides: []int{0, 0, 0}, Offset: 2}
	out := testAlloc(t, b, 30)
	if err := b.Gather(out, expanded); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 30)
	for i := range want {
		want[i] = data[2]
	}
	checkValues(t, b, out, want)
	if err := b.Scatter(expanded, out); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.Gather(base, gpu.View{Buffer: base, Shape: []int{24}, Strides: []int{1}}); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.Gather(out, gpu.View{Buffer: base, Shape: []int{30}, Strides: []int{1}}); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if !b.Available() {
		t.Fatal("bad view disabled healthy device")
	}
}

func TestVulkanViewGEMM(t *testing.T) {
	b := testBackend(t)
	for _, tc := range []struct {
		as, ast, bs, bst []int
		ao, bo, an, bn   int
	}{
		{[]int{2, 1, 3, 5}, []int{15, 15, 5, 1}, []int{1, 4, 5, 7}, []int{140, 35, 7, 1}, 0, 0, 30, 140},
		{[]int{2, 3, 5}, []int{35, 1, 7}, []int{1, 5, 4}, []int{20, 1, 5}, 1, 0, 70, 20},
		{[]int{2, 3}, []int{0, 0}, []int{3, 4}, []int{4, 1}, 0, 0, 1, 12},
		{[]int{1, 2, 1, 1, 3, 5}, []int{30, 15, 15, 15, 5, 1}, []int{3, 1, 4, 1, 5, 7}, []int{140, 140, 35, 35, 7, 1}, 0, 0, 30, 420},
	} {
		t.Run(fmt.Sprintf("%v_%v", tc.as, tc.bs), func(t *testing.T) {
			a, right := make([]float32, tc.an), make([]float32, tc.bn)
			for i := range a {
				a[i] = float32(i%17-8) / 16
			}
			for i := range right {
				right[i] = float32(i%13-6) / 8
			}
			av := gpu.View{Buffer: testUpload(t, b, a), Shape: tc.as, Strides: tc.ast, Offset: tc.ao}
			bv := gpu.View{Buffer: testUpload(t, b, right), Shape: tc.bs, Strides: tc.bst, Offset: tc.bo}
			shape, err := gpu.MatMulShape(tc.as, tc.bs)
			if err != nil {
				t.Fatal(err)
			}
			n, err := gpu.Elements(shape...)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, n)
			m, k, cols := tc.as[len(tc.as)-2], tc.as[len(tc.as)-1], tc.bs[len(tc.bs)-1]
			batch := shape[:len(shape)-2]
			batches, _ := gpu.Elements(batch...)
			batchOffset := func(z int, v gpu.View) int {
				offset, rest, pad := v.Offset, z, len(batch)-(len(v.Shape)-2)
				for i := len(batch) - 1; i >= 0; i-- {
					coord := rest % batch[i]
					rest /= batch[i]
					if j := i - pad; j >= 0 && v.Shape[j] != 1 {
						offset += coord * v.Strides[j]
					}
				}
				return offset
			}
			for z := 0; z < batches; z++ {
				ao, bo := batchOffset(z, av), batchOffset(z, bv)
				for i := 0; i < m; i++ {
					for j := 0; j < cols; j++ {
						var sum float32
						for d := 0; d < k; d++ {
							sum += a[ao+i*av.Strides[len(tc.as)-2]+d*av.Strides[len(tc.as)-1]] * right[bo+d*bv.Strides[len(tc.bs)-2]+j*bv.Strides[len(tc.bs)-1]]
						}
						want[(z*m+i)*cols+j] = sum
					}
				}
			}
			out := testAlloc(t, b, n)
			if err := b.MatMulViews(out, av, bv); err != nil {
				t.Fatal(err)
			}
			checkMatrix(t, b, out, want)
		})
	}
}
