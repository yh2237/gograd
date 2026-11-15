package autograd

import (
	"fmt"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

// BenchmarkTransformerTrainingStepCPU measures a full encoder training step:
// forward, backward, gradient clip and AdamW.
func BenchmarkTransformerTrainingStepCPU(b *testing.B) {
	const (
		model  = 256
		heads  = 4
		ffn    = 1024
		batch  = 8
		seq    = 64
		layers = 2
	)
	root := &Module{}
	stack := make([]*TransformerEncoderLayer, layers)
	for i := range stack {
		layer, err := NewTransformerEncoderLayer(model, heads, ffn, tensor.CPU)
		if err != nil {
			b.Fatal(err)
		}
		stack[i] = layer
		root.Children = append(root.Children, NamedModule{Name: fmt.Sprintf("l%d", i), Module: &layer.Module})
	}
	params := root.NamedParameters()
	opt := NewAdamW(params, .001, .01)
	x, err := New(make([]float32, batch*seq*model), []int{batch, seq, model}, tensor.CPU, false)
	if err != nil {
		b.Fatal(err)
	}
	defer x.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.ZeroGrad()
		y := x
		for _, layer := range stack {
			y = layer.Forward(y, nil, uint32(i))
		}
		loss := Mean(Mul(y, y), 0, 1, 2)
		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}
		ClipGradNorm(params, 1)
		opt.Step()
		loss.ReleaseGraph()
	}
}
