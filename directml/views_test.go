package directml

import (
	"errors"
	"fmt"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func viewIndex(flat int, v gpu.View) int {
	index := v.Offset
	for i := len(v.Shape) - 1; i >= 0; i-- {
		index += flat % v.Shape[i] * v.Strides[i]
		flat /= v.Shape[i]
	}
	return index
}
func TestDirectMLGatherScatter(t *testing.T) {
	b := testBackend(t)
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	base := upload(t, b, data)
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
		out := alloc(t, b, n)
		want := make([]float32, n)
		for i := range want {
			want[i] = data[viewIndex(i, v)]
		}
		if err := b.Gather(out, v); err != nil {
			t.Fatal(err)
		}
		check(t, b, out, want)
		for i := range want {
			want[i] = float32(100 + i)
			data[viewIndex(i, v)] = want[i]
		}
		if err := b.Scatter(v, upload(t, b, want)); err != nil {
			t.Fatal(err)
		}
		check(t, b, base, data)
	}
	v := gpu.View{Buffer: base, Shape: []int{2, 3, 5}, Strides: []int{0, 0, 0}, Offset: 2}
	out := alloc(t, b, 30)
	if err := b.Gather(out, v); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 30)
	for i := range want {
		want[i] = data[2]
	}
	check(t, b, out, want)
	if err := b.Scatter(v, out); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if err := b.Gather(base, gpu.View{Buffer: base, Shape: []int{24}, Strides: []int{1}}); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if !b.Available() {
		t.Fatal("bad view disabled healthy device")
	}
}

func TestDirectMLViewGEMM(t *testing.T) {
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
			av, bv := gpu.View{Buffer: upload(t, b, a), Shape: tc.as, Strides: tc.ast, Offset: tc.ao}, gpu.View{Buffer: upload(t, b, right), Shape: tc.bs, Strides: tc.bst, Offset: tc.bo}
			shape, err := gpu.MatMulShape(tc.as, tc.bs)
			if err != nil {
				t.Fatal(err)
			}
			n, err := gpu.Elements(shape...)
			if err != nil {
				t.Fatal(err)
			}
			m, k, cols := tc.as[len(tc.as)-2], tc.as[len(tc.as)-1], tc.bs[len(tc.bs)-1]
			batch := shape[:len(shape)-2]
			batches, _ := gpu.Elements(batch...)
			av, err = av.BroadcastTo(append(append([]int(nil), batch...), m, k))
			if err != nil {
				t.Fatal(err)
			}
			bv, err = bv.BroadcastTo(append(append([]int(nil), batch...), k, cols))
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, n)
			for z := 0; z < batches; z++ {
				for i := 0; i < m; i++ {
					for j := 0; j < cols; j++ {
						var sum float32
						for d := 0; d < k; d++ {
							sum += a[viewIndex((z*m+i)*k+d, av)] * right[viewIndex((z*k+d)*cols+j, bv)]
						}
						want[(z*m+i)*cols+j] = sum
					}
				}
			}
			out := alloc(t, b, n)
			if err := b.MatMulViews(out, av, bv); err != nil {
				t.Fatal(err)
			}
			checkTransformer(t, b, out, want)
		})
	}
}
