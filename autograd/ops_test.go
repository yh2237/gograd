package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"testing"
)

func TestBroadcastAndMatmulGrad(t *testing.T) {
	a := Must([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)
	b := Must([]float32{2, 3, 4}, []int{3}, true)
	loss := Sum(Mul(a, b), 0, 1)
	if e := loss.Backward(); e != nil {
		t.Fatal(e)
	}
	closeValues(t, "broadcast a", a.Grad, []float32{2, 3, 4, 2, 3, 4}, 0)
	closeValues(t, "broadcast b", b.Grad, []float32{5, 7, 9}, 0)
	x := Must([]float32{1, 2, 3, 4}, []int{2, 2}, true)
	w := Must([]float32{5, 6, 7, 8}, []int{2, 2}, true)
	out := Sum(MatMul(x, w), 0, 1)
	if e := out.Backward(); e != nil {
		t.Fatal(e)
	}
	closeValues(t, "matmul x", x.Grad, []float32{11, 15, 11, 15}, 0)
	closeValues(t, "matmul w", w.Grad, []float32{4, 4, 6, 6}, 0)
}
func TestCUDAOpsResident(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	context, e := NewCUDAContext()
	if e != nil {
		t.Fatal(e)
	}
	defer context.Close()
	a, e := New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, tensor.CUDA, true)
	if e != nil {
		t.Fatal(e)
	}
	b, e := New([]float32{2, 3, 4}, []int{3}, tensor.CUDA, true)
	if e != nil {
		t.Fatal(e)
	}
	loss := Sum(Mul(a, b), 0, 1)
	if loss.Buffer() == nil || loss.Data != nil {
		t.Fatal("reduction staged on host")
	}
	v, e := loss.ToHost()
	if e != nil {
		t.Fatal(e)
	}
	closeValues(t, "loss", v, []float32{67}, 1e-5)
	if e = loss.Backward(); e != nil {
		t.Fatal(e)
	}
	ag, e := a.GradToHost()
	if e != nil {
		t.Fatal(e)
	}
	bg, e := b.GradToHost()
	if e != nil {
		t.Fatal(e)
	}
	closeValues(t, "a", ag, []float32{2, 3, 4, 2, 3, 4}, 1e-5)
	closeValues(t, "b", bg, []float32{5, 7, 9}, 1e-5)
}
func TestShapeEmbeddingAndScheduler(t *testing.T) {
	p := Must([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, true)
	v := Embedding(p, []int{1, 1, 2}, []int{3})
	s := Sum(Slice(v, 0, 0, 2), 0, 1)
	if e := s.Backward(); e != nil {
		t.Fatal(e)
	}
	closeValues(t, "embedding", p.Grad, []float32{0, 0, 2, 2, 0, 0}, 0)
	c := NewOneCycle(.01, 10, .3)
	if math.Abs(c.LR()-.0004) > 1e-8 {
		t.Fatal(c.LR())
	}
	for i := 0; i < 2; i++ {
		c.StepCount++
	}
	if math.Abs(c.LR()-.01) > 1e-8 {
		t.Fatal(c.LR())
	}
}
func TestModuleStateDict(t *testing.T) {
	p := Must([]float32{1, 2}, []int{2}, true)
	m := &Module{Children: []NamedModule{{Name: "layer", Module: &Module{Parameters: []Parameter{{Name: "weight", Value: p}}}}}}
	state := m.StateDict()
	state["layer.weight"][0] = 7
	if p.Data[0] != 1 {
		t.Fatal("state aliases parameter")
	}
	if e := m.LoadStateDict(state); e != nil {
		t.Fatal(e)
	}
	if p.Data[0] != 7 {
		t.Fatal("load did not update parameter")
	}
	if e := m.LoadStateDict(map[string][]float32{}); e == nil {
		t.Fatal("expected state mismatch")
	}
}
