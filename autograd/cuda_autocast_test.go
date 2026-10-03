package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestBF16AutocastCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	aData := make([]float32, 2*16*32)
	bData := make([]float32, 32*24)
	for i := range aData {
		aData[i] = float32(math.Sin(float64(i))) * .2
	}
	for i := range bData {
		bData[i] = float32(math.Cos(float64(i))) * .2
	}
	run := func(bf16 bool) ([]float32, []float32, []float32) {
		BF16Autocast = bf16
		a, _ := New(aData, []int{2, 16, 32}, tensor.CUDA, true)
		b, _ := New(bData, []int{32, 24}, tensor.CUDA, true)
		y := MatMul(a, b)
		v, e := y.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		if e := Mean(Mul(y, y), 0, 1, 2).Backward(); e != nil {
			t.Fatal(e)
		}
		da, _ := a.GradToHost()
		db, _ := b.GradToHost()
		y.ReleaseGraph()
		a.Close()
		b.Close()
		return v, da, db
	}
	defer func() { BF16Autocast = false }()
	f, fa, fb := run(false)
	EnableGPUProfile(true)
	g, ga, gb := run(true)
	p := GPUProfile()
	EnableGPUProfile(false)
	if p["f32_to_bf16"].Count == 0 {
		t.Fatal("BF16 conversion did not run on CUDA")
	}
	for _, pair := range [][2][]float32{{f, g}, {fa, ga}, {fb, gb}} {
		for i := range pair[0] {
			if math.Abs(float64(pair[0][i]-pair[1][i])) > .02 {
				t.Fatalf("BF16 mismatch at %d: %g vs %g", i, pair[0][i], pair[1][i])
			}
		}
	}
}

func TestBF16GraphCaptureCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	BF16Autocast = true
	defer func() { BF16Autocast = false }()
	t.Run("acoustic", TestAcousticGraphCaptureCUDA)
	t.Run("transformer", TestTransformerGraphCaptureCUDA)
}
