package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestAcousticGraphCaptureCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, e := NewCUDAContext()
	if e != nil {
		t.Skip(e)
	}
	defer ctx.Close()
	graphModel, e := NewAcoustic(8, 3, 4, []int{1}, tensor.CUDA)
	if e != nil {
		t.Fatal(e)
	}
	eagerModel, e := NewAcoustic(8, 3, 4, []int{1}, tensor.CUDA)
	if e != nil {
		t.Fatal(e)
	}
	rng := rand.New(rand.NewSource(405))
	for i, p := range graphModel.Params {
		values := make([]float32, p.Value.Numel())
		for j := range values {
			values[j] = float32(rng.NormFloat64() * .03)
		}
		if e := p.Value.CopyFrom(values); e != nil {
			t.Fatal(e)
		}
		if e := eagerModel.Params[i].Value.CopyFrom(values); e != nil {
			t.Fatal(e)
		}
	}
	ids := make([]int, 2*5*3)
	for i := range ids {
		ids[i] = i % 46
	}
	speaker := []int{0, 2}
	for _, m := range []*Acoustic{graphModel, eagerModel} {
		if e := m.SetCUDAIndices(ids, speaker); e != nil {
			t.Fatal(e)
		}
		defer m.ClearCUDAIndices()
	}
	cont, e := New(make([]float32, 2*5*4), []int{2, 5, 4}, tensor.CUDA, false)
	if e != nil {
		t.Fatal(e)
	}
	defer cont.Close()
	target, e := New(make([]float32, 2*5*4), []int{2, 5, 4}, tensor.CUDA, false)
	if e != nil {
		t.Fatal(e)
	}
	defer target.Close()
	graphOpt := NewAdamW(graphModel.Params, .001, .01)
	eagerOpt := NewAdamW(eagerModel.Params, .001, .01)
	step := func(m *Acoustic, opt *AdamW) error {
		opt.ZeroGrad()
		loss := MaskedLoss(m.Forward(ids, cont, speaker), target, false)
		if e := loss.Backward(); e != nil {
			return e
		}
		ClipGradNorm(m.Params, 1)
		opt.Step()
		loss.ReleaseGraph()
		return nil
	}
	if e := step(graphModel, graphOpt); e != nil {
		t.Fatal(e)
	}
	if e := step(eagerModel, eagerOpt); e != nil {
		t.Fatal(e)
	}
	if e := graphOpt.PrepareGraph(); e != nil {
		t.Fatal(e)
	}
	stream, e := cuda.NewStream()
	if e != nil {
		t.Fatal(e)
	}
	defer stream.Destroy()
	if e := SetBLASStream(stream); e != nil {
		t.Fatal(e)
	}
	graph, e := cuda.Capture(stream, func() error { return step(graphModel, graphOpt) })
	if e != nil {
		t.Fatal(e)
	}
	defer graph.Close()
	for i := 0; i < 2; i++ {
		if e := graph.Launch(); e != nil {
			t.Fatal(e)
		}
	}
	if e := stream.Synchronize(); e != nil {
		t.Fatal(e)
	}
	if e := SetBLASStream(nil); e != nil {
		t.Fatal(e)
	}
	count, e := graphOpt.GraphStepCount()
	if e != nil {
		t.Fatal(e)
	}
	if count != 3 {
		t.Fatalf("device optimizer step %d, want 3", count)
	}
	for i := 0; i < 2; i++ {
		if e := step(eagerModel, eagerOpt); e != nil {
			t.Fatal(e)
		}
	}
	for i, p := range graphModel.Params {
		got, e := p.Value.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		want, e := eagerModel.Params[i].Value.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		for j := range got {
			if math.Abs(float64(got[j]-want[j])) > 2e-4 {
				t.Fatalf("%s[%d] graph %g eager %g", p.Name, j, got[j], want[j])
			}
		}
	}
}
