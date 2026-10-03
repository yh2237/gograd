package autograd

import (
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestFlashAttentionParityCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	qdata, kdata, vdata, mdata, updata := make([]float32, 2*3*4*8), make([]float32, 2*3*5*8), make([]float32, 2*3*5*6), make([]float32, 1*1*4*5), make([]float32, 2*3*4*6)
	for i := range qdata {
		qdata[i] = float32(math.Sin(float64(i)*.17)) * .3
	}
	for i := range kdata {
		kdata[i] = float32(math.Cos(float64(i)*.11)) * .3
	}
	for i := range vdata {
		vdata[i] = float32(math.Sin(float64(i)*.23)) * .3
	}
	for i := range mdata {
		mdata[i] = float32(math.Cos(float64(i)*.29)) * .1
	}
	for i := range updata {
		updata[i] = float32(math.Sin(float64(i)*.31)) * .2
	}
	run := func(mode string) [][]float32 {
		AttentionAlgorithm = mode
		q, _ := New(qdata, []int{2, 3, 4, 8}, tensor.CUDA, true)
		k, _ := New(kdata, []int{2, 3, 5, 8}, tensor.CUDA, true)
		v, _ := New(vdata, []int{2, 3, 5, 6}, tensor.CUDA, true)
		mask, _ := New(mdata, []int{1, 1, 4, 5}, tensor.CUDA, true)
		up, _ := New(updata, []int{2, 3, 4, 6}, tensor.CUDA, false)
		y := ScaledDotProductAttention(q, k, v, mask)
		out, e := y.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		if e := weightedSum(y, up).Backward(); e != nil {
			t.Fatal(e)
		}
		qg, _ := q.GradToHost()
		kg, _ := k.GradToHost()
		vg, _ := v.GradToHost()
		mg, _ := mask.GradToHost()
		y.ReleaseGraph()
		q.Close()
		k.Close()
		v.Close()
		mask.Close()
		up.Close()
		return [][]float32{out, qg, kg, vg, mg}
	}
	defer func() { AttentionAlgorithm = "auto" }()
	want := run("materialized")
	EnableGPUProfile(true)
	got := run("flash")
	profile := GPUProfile()
	EnableGPUProfile(false)
	if profile["flash_attention_tiled_f"].Count == 0 || profile["flash_attention_tiled_b"].Count == 0 {
		t.Fatal("FlashAttention kernels did not run on device")
	}
	for j := range want {
		for i := range want[j] {
			if math.Abs(float64(want[j][i]-got[j][i])) > 2e-3 {
				t.Fatalf("component %d[%d]: materialized %g flash %g", j, i, want[j][i], got[j][i])
			}
		}
	}
}

func TestFlashAttentionSequenceParityCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	defer func() { AttentionAlgorithm = "auto" }()
	for _, seq := range []int{256, 1024, 4096} {
		t.Run(fmt.Sprint(seq), func(t *testing.T) {
			ctx, err := NewCUDAContext()
			if err != nil {
				t.Skip(err)
			}
			defer ctx.Close()
			shape := []int{1, 1, seq, 16}
			size := seq * 16
			data := func(phase float64) []float32 {
				x := make([]float32, size)
				for i := range x {
					x[i] = float32(math.Sin(float64(i)*.013+phase)) * .1
				}
				return x
			}
			qd, kd, vd, ud := data(.1), data(.4), data(.7), data(1.0)
			run := func(mode string) [][]float32 {
				AttentionAlgorithm = mode
				q, _ := New(qd, shape, tensor.CUDA, true)
				k, _ := New(kd, shape, tensor.CUDA, true)
				v, _ := New(vd, shape, tensor.CUDA, true)
				up, _ := New(ud, shape, tensor.CUDA, false)
				y := ScaledDotProductAttention(q, k, v, nil)
				out, _ := y.ToHost()
				if e := weightedSum(y, up).Backward(); e != nil {
					t.Fatal(e)
				}
				qg, _ := q.GradToHost()
				kg, _ := k.GradToHost()
				vg, _ := v.GradToHost()
				y.ReleaseGraph()
				q.Close()
				k.Close()
				v.Close()
				up.Close()
				return [][]float32{out, qg, kg, vg}
			}
			want, got := run("materialized"), run("flash")
			for j := range want {
				for i := range want[j] {
					if math.Abs(float64(want[j][i]-got[j][i])) > 3e-3 {
						t.Fatalf("seq %d component %d[%d]: %g vs %g", seq, j, i, want[j][i], got[j][i])
					}
				}
			}
		})
	}
}
