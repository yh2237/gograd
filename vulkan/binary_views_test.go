package vulkan

import (
	"math/rand/v2"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

// viewElements reads the logical elements of a view from its backing data.
func viewElements(data []float32, v gpu.View) []float32 {
	n, err := gpu.Elements(v.Shape...)
	if err != nil {
		panic(err)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = data[viewIndex(i, v.Shape, v.Strides, v.Offset)]
	}
	return out
}

// TestVulkanBinaryViews checks the elementwise view operations against a host
// reference. The operands keep their strides, so a broadcast or transposed
// view must not be expanded first.
func TestVulkanBinaryViews(t *testing.T) {
	b := testBackend(t)
	rng := rand.New(rand.NewPCG(97, 31))
	for _, tc := range []struct {
		name     string
		aShape   []int
		aStrides []int
		aOffset  int
		bShape   []int
		bStrides []int
		bOffset  int
	}{
		{name: "contiguous", aShape: []int{2, 3}, aStrides: []int{3, 1}, bShape: []int{2, 3}, bStrides: []int{3, 1}},
		{name: "broadcast_row", aShape: []int{2, 3}, aStrides: []int{3, 1}, bShape: []int{2, 3}, bStrides: []int{0, 1}},
		{name: "broadcast_scalar", aShape: []int{2, 3}, aStrides: []int{3, 1}, bShape: []int{2, 3}, bStrides: []int{0, 0}},
		{name: "transposed", aShape: []int{2, 3}, aStrides: []int{1, 3}, bShape: []int{2, 3}, bStrides: []int{1, 3}},
		{name: "offset", aShape: []int{2, 3}, aStrides: []int{3, 1}, aOffset: 2, bShape: []int{2, 3}, bStrides: []int{3, 1}, bOffset: 1},
		{name: "rank3", aShape: []int{2, 2, 3}, aStrides: []int{6, 3, 1}, bShape: []int{2, 2, 3}, bStrides: []int{6, 3, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			aData := make([]float32, 32)
			bData := make([]float32, 32)
			for i := range aData {
				aData[i] = rng.Float32() - .5
			}
			for i := range bData {
				bData[i] = rng.Float32() - .5
			}
			aView := gpu.View{Buffer: testUpload(t, b, aData), Shape: tc.aShape, Strides: tc.aStrides, Offset: tc.aOffset}
			bView := gpu.View{Buffer: testUpload(t, b, bData), Shape: tc.bShape, Strides: tc.bStrides, Offset: tc.bOffset}
			aElements := viewElements(aData, aView)
			bElements := viewElements(bData, bView)
			n, err := gpu.Elements(tc.aShape...)
			if err != nil {
				t.Fatal(err)
			}
			wantAdd := make([]float32, n)
			wantMul := make([]float32, n)
			for i := range wantAdd {
				wantAdd[i] = aElements[i] + bElements[i]
				wantMul[i] = aElements[i] * bElements[i]
			}
			out := testAlloc(t, b, n)
			if err := b.AddViews(out, aView, bView); err != nil {
				t.Fatal(err)
			}
			checkValues(t, b, out, wantAdd)
			if err := b.MulViews(out, aView, bView); err != nil {
				t.Fatal(err)
			}
			checkValues(t, b, out, wantMul)
		})
	}
	if s := b.MemoryStats(); s.LiveBuffers != 0 {
		t.Fatalf("binary views leaked: %+v", s)
	}
}

// TestVulkanBinaryViewsRejectsMismatch checks that operands with different
// shapes are rejected instead of computing a wrong result.
func TestVulkanBinaryViewsRejectsMismatch(t *testing.T) {
	b := testBackend(t)
	data := make([]float32, 12)
	for i := range data {
		data[i] = float32(i)
	}
	buffer := testUpload(t, b, data)
	out := testAlloc(t, b, 6)
	left := gpu.View{Buffer: buffer, Shape: []int{2, 3}, Strides: []int{3, 1}}
	right := gpu.View{Buffer: buffer, Shape: []int{3, 2}, Strides: []int{2, 1}}
	if err := b.AddViews(out, left, right); err == nil {
		t.Fatal("mismatched view shapes were accepted")
	}
}

// TestVulkanGatherContiguousView checks the copy path of a compact view, which
// replaces the index arithmetic of the view shader.
func TestVulkanGatherContiguousView(t *testing.T) {
	b := testBackend(t)
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	base := testUpload(t, b, data)
	for _, v := range []gpu.View{
		{Buffer: base, Shape: []int{2, 3}, Strides: []int{3, 1}},
		{Buffer: base, Shape: []int{2, 3}, Strides: []int{3, 1}, Offset: 4},
		{Buffer: base, Shape: []int{2, 1, 3}, Strides: []int{3, 0, 1}},
		{Buffer: base, Shape: []int{4}, Strides: []int{1}, Offset: 6},
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
	}
}
