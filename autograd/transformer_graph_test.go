package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestTransformerGraphCaptureCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, e := NewCUDAContext()
	if e != nil {
		t.Skip(e)
	}
	defer ctx.Close()
	makeModel := func() ([]*TransformerEncoderLayer, []Parameter) {
		layers := make([]*TransformerEncoderLayer, 2)
		root := &Module{}
		for i := range layers {
			layer, e := NewTransformerEncoderLayer(8, 2, 16, tensor.CUDA)
			if e != nil {
				t.Fatal(e)
			}
			layers[i] = layer
			root.Children = append(root.Children, NamedModule{Name: string(rune('0' + i)), Module: &layer.Module})
		}
		return layers, root.NamedParameters()
	}
	graphLayers, graphParams := makeModel()
	eagerLayers, eagerParams := makeModel()
	x, e := New(make([]float32, 2*4*8), []int{2, 4, 8}, tensor.CUDA, false)
	if e != nil {
		t.Fatal(e)
	}
	defer x.Close()
	graphOpt := NewAdamW(graphParams, .001, .01)
	eagerOpt := NewAdamW(eagerParams, .001, .01)
	step := func(layers []*TransformerEncoderLayer, params []Parameter, opt *AdamW) error {
		opt.ZeroGrad()
		y := x
		for _, layer := range layers {
			y = layer.Forward(y, nil, 1)
		}
		loss := Mean(Mul(y, y), 0, 1, 2)
		if e := loss.Backward(); e != nil {
			return e
		}
		ClipGradNorm(params, 1)
		opt.Step()
		loss.ReleaseGraph()
		return nil
	}
	if e := step(graphLayers, graphParams, graphOpt); e != nil {
		t.Fatal(e)
	}
	if e := step(eagerLayers, eagerParams, eagerOpt); e != nil {
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
	graph, e := cuda.Capture(stream, func() error { return step(graphLayers, graphParams, graphOpt) })
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
		t.Fatalf("device step %d, want 3", count)
	}
	for i := 0; i < 2; i++ {
		if e := step(eagerLayers, eagerParams, eagerOpt); e != nil {
			t.Fatal(e)
		}
	}
	for i, p := range graphParams {
		got, e := p.Value.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		want, e := eagerParams[i].Value.ToHost()
		if e != nil {
			t.Fatal(e)
		}
		for j := range got {
			if math.Abs(float64(got[j]-want[j])) > 3e-4 {
				t.Fatalf("%s[%d]: graph %g eager %g", p.Name, j, got[j], want[j])
			}
		}
	}
}
