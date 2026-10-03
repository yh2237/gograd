package autograd

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/yh2237/gograd/tensor"
)

// MultiheadAttention uses batch-first [batch, sequence, model] tensors and
// PyTorch's packed in_proj_weight/in_proj_bias parameter layout.
type MultiheadAttention struct {
	Module
	Model, Heads int
	Dropout      float32
	Training     bool
}

func newParameter(shape []int, device tensor.Device, seed int64, scale float32) (*Tensor, error) {
	data := make([]float32, numel(shape))
	rng := rand.New(rand.NewSource(seed))
	for i := range data {
		data[i] = (rng.Float32()*2 - 1) * scale
	}
	return New(data, shape, device, true)
}
func NewMultiheadAttention(model, heads int, device tensor.Device) (*MultiheadAttention, error) {
	if model < 1 || heads < 1 || model%heads != 0 {
		return nil, fmt.Errorf("autograd: invalid attention dimensions")
	}
	m := &MultiheadAttention{Model: model, Heads: heads, Training: true}
	for i, p := range []struct {
		name  string
		shape []int
		scale float32
	}{
		{"in_proj_weight", []int{3 * model, model}, float32(math.Sqrt(1 / float64(model)))},
		{"in_proj_bias", []int{3 * model}, 0},
		{"out_proj.weight", []int{model, model}, float32(math.Sqrt(1 / float64(model)))},
		{"out_proj.bias", []int{model}, 0},
	} {
		v, e := newParameter(p.shape, device, int64(i+1), p.scale)
		if e != nil {
			return nil, e
		}
		m.Parameters = append(m.Parameters, Parameter{p.name, v})
	}
	return m, nil
}
func (m *MultiheadAttention) param(name string) *Tensor {
	for _, p := range m.Parameters {
		if p.Name == name {
			return p.Value
		}
	}
	panic("autograd: missing attention parameter")
}
func (m *MultiheadAttention) Forward(q, k, v, mask *Tensor) *Tensor {
	return m.ForwardSeed(q, k, v, mask, 0)
}
func (m *MultiheadAttention) ForwardSeed(q, k, v, mask *Tensor, seed uint32) *Tensor {
	if len(q.Shape) != 3 || len(k.Shape) != 3 || len(v.Shape) != 3 || q.Shape[0] != k.Shape[0] || k.Shape[0] != v.Shape[0] || k.Shape[1] != v.Shape[1] || q.Shape[2] != m.Model || k.Shape[2] != m.Model || v.Shape[2] != m.Model {
		panic("autograd: attention shape")
	}
	b, t, source := q.Shape[0], q.Shape[1], k.Shape[1]
	weight, bias := m.param("in_proj_weight"), m.param("in_proj_bias")
	project := func(x *Tensor, start int) *Tensor {
		w := Slice(weight, 0, start, start+m.Model)
		z := MatMul(x, Transpose(w, 0, 1))
		return Add(z, Slice(bias, 0, start, start+m.Model))
	}
	qh, kh, vh := project(q, 0), project(k, m.Model), project(v, 2*m.Model)
	toHeads := func(x *Tensor, seq int) *Tensor {
		return Permute(Reshape(x, b, seq, m.Heads, m.Model/m.Heads), 0, 2, 1, 3)
	}
	att := attentionWithDropout(toHeads(qh, t), toHeads(kh, source), toHeads(vh, source), mask, m.Dropout, seed, m.Training)
	merged := Reshape(Permute(att, 0, 2, 1, 3), b, t, m.Model)
	return Add(MatMul(merged, Transpose(m.param("out_proj.weight"), 0, 1)), m.param("out_proj.bias"))
}

// TransformerEncoderLayer is the pre-norm, batch-first encoder variant.
type TransformerEncoderLayer struct {
	Module
	Attention          *MultiheadAttention
	Model, FeedForward int
	Dropout            float32
	Training           bool
}

func NewTransformerEncoderLayer(model, heads, feedForward int, device tensor.Device) (*TransformerEncoderLayer, error) {
	att, e := NewMultiheadAttention(model, heads, device)
	if e != nil {
		return nil, e
	}
	m := &TransformerEncoderLayer{Attention: att, Model: model, FeedForward: feedForward, Training: true}
	m.Children = []NamedModule{{"self_attn", &att.Module}}
	for i, p := range []struct {
		name  string
		shape []int
		scale float32
	}{
		{"norm1.weight", []int{model}, 0}, {"norm1.bias", []int{model}, 0},
		{"linear1.weight", []int{feedForward, model}, float32(math.Sqrt(1 / float64(model)))}, {"linear1.bias", []int{feedForward}, 0},
		{"linear2.weight", []int{model, feedForward}, float32(math.Sqrt(1 / float64(feedForward)))}, {"linear2.bias", []int{model}, 0},
		{"norm2.weight", []int{model}, 0}, {"norm2.bias", []int{model}, 0},
	} {
		v, e := newParameter(p.shape, device, int64(100+i), p.scale)
		if e != nil {
			return nil, e
		}
		if p.name == "norm1.weight" || p.name == "norm2.weight" {
			ones := make([]float32, model)
			for j := range ones {
				ones[j] = 1
			}
			if e := v.CopyFrom(ones); e != nil {
				return nil, e
			}
		}
		m.Parameters = append(m.Parameters, Parameter{p.name, v})
	}
	return m, nil
}
func (m *TransformerEncoderLayer) param(name string) *Tensor {
	for _, p := range m.Parameters {
		if p.Name == name {
			return p.Value
		}
	}
	panic("autograd: missing encoder parameter")
}
func (m *TransformerEncoderLayer) Forward(x, mask *Tensor, stepSeed uint32) *Tensor {
	h := LayerNorm(x, m.param("norm1.weight"), m.param("norm1.bias"), 1e-5)
	h = m.Attention.ForwardSeed(h, h, h, mask, stepSeed+3)
	x = Add(x, Dropout(h, m.Dropout, stepSeed, m.Training))
	h = LayerNorm(x, m.param("norm2.weight"), m.param("norm2.bias"), 1e-5)
	h = BiasGELU(MatMul(h, Transpose(m.param("linear1.weight"), 0, 1)), m.param("linear1.bias"))
	h = Dropout(h, m.Dropout, stepSeed+1, m.Training)
	h = MatMul(h, Transpose(m.param("linear2.weight"), 0, 1))
	if !m.Training || m.Dropout == 0 {
		return BiasResidual(h, m.param("linear2.bias"), x)
	}
	h = Add(h, m.param("linear2.bias"))
	return Add(x, Dropout(h, m.Dropout, stepSeed+2, true))
}
