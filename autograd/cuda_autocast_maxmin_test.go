package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestBF16AutocastMaxMinCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	aData := make([]float32, 64)
	bData := make([]float32, 64)
	for i := range aData {
		aData[i] = float32(math.Sin(float64(i)*.7)) * 2
		bData[i] = float32(math.Cos(float64(i)*.5)) * 2
	}
	run := func(bf16 bool, op func(*Tensor, *Tensor) *Tensor) ([]float32, []float32, []float32) {
		BF16Autocast = bf16
		a, _ := New(aData, []int{8, 8}, tensor.CUDA, true)
		b, _ := New(bData, []int{8, 8}, tensor.CUDA, true)
		y := op(a, b)
		v, e := y.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		if e := Sum(y, 0, 1).Backward(); e != nil {
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
	for _, tc := range []struct {
		name string
		op   func(*Tensor, *Tensor) *Tensor
	}{
		{"max", Max},
		{"min", Min},
	} {
		f, fa, fb := run(false, tc.op)
		g, ga, gb := run(true, tc.op)
		for i := range f {
			if math.Abs(float64(g[i]-f[i])) > 1e-2*math.Max(1, math.Abs(float64(f[i]))) {
				t.Fatalf("%s value[%d]: bf16 %g, fp32 %g", tc.name, i, g[i], f[i])
			}
			if ga[i] != fa[i] || gb[i] != fb[i] {
				t.Fatalf("%s gradient[%d]: bf16 (%g,%g), fp32 (%g,%g)", tc.name, i, ga[i], gb[i], fa[i], fb[i])
			}
			if fa[i]+fb[i] != 1 {
				t.Fatalf("%s gradient[%d] sums to %g, want 1", tc.name, i, fa[i]+fb[i])
			}
		}
	}
}
