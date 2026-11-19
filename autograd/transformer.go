package autograd

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/yh2237/gograd/gpu"
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
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	data := make([]float32, numel(shape))
	rng := rand.New(rand.NewSource(seed))
	for i := range data {
		data[i] = (rng.Float32()*2 - 1) * scale
	}
	return New(data, shape, device, true)
}
func closeParameters(params []Parameter) {
	for _, p := range params {
		if p.Value != nil {
			p.Value.Close()
		}
	}
}

func NewMultiheadAttention(model, heads int, device tensor.Device) (*MultiheadAttention, error) {
	if model < 1 || heads < 1 || model%heads != 0 {
		return nil, fmt.Errorf("autograd: invalid attention dimensions")
	}
	m := &MultiheadAttention{Model: model, Heads: heads, Training: true}
	m.Module.Training = true
	m.Module.OnTrainingChange = func(training bool) { m.Training = training }
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
			closeParameters(m.Parameters)
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
func cpuSplitHeads(x *Tensor, b, t, heads, dim, parts, which int) *Tensor {
	x = x.Contiguous()
	out := cpuAllocUninit(b * heads * t * dim)
	model := heads * dim
	stride := parts * model
	offset := which * model
	parallelForWork(b*heads*t, dim, func(start, end int) {
		for idx := start; idx < end; idx++ {
			ti := idx % t
			tmp := idx / t
			h := tmp % heads
			bi := tmp / heads
			src := (bi*t+ti)*stride + offset + h*dim
			copy(out[idx*dim:(idx+1)*dim], x.Data[src:src+dim])
		}
	})
	return result(out, []int{b, heads, t, dim}, []*Tensor{x}, func(g []float32) {
		if !x.RequiresGrad {
			return
		}
		if x.Grad == nil {
			x.Grad = make([]float32, x.Numel())
		}
		grad := x.Grad
		parallelForWork(b*heads*t, dim, func(start, end int) {
			for idx := start; idx < end; idx++ {
				ti := idx % t
				tmp := idx / t
				h := tmp % heads
				bi := tmp / heads
				dst := (bi*t+ti)*stride + offset + h*dim
				src := idx * dim
				for d := 0; d < dim; d++ {
					grad[dst+d] += g[src+d]
				}
			}
		})
	})
}

func cpuMergeHeads(x *Tensor, b, t, heads, dim int) *Tensor {
	x = x.Contiguous()
	out := cpuAllocUninit(b * t * heads * dim)
	model := heads * dim
	parallelForWork(b*t*heads, dim, func(start, end int) {
		for idx := start; idx < end; idx++ {
			h := idx % heads
			tmp := idx / heads
			ti := tmp % t
			bi := tmp / t
			src := ((bi*heads+h)*t + ti) * dim
			dst := (bi*t+ti)*model + h*dim
			copy(out[dst:dst+dim], x.Data[src:src+dim])
		}
	})
	return result(out, []int{b, t, model}, []*Tensor{x}, func(g []float32) {
		if !x.RequiresGrad {
			return
		}
		if x.Grad == nil {
			x.Grad = make([]float32, x.Numel())
		}
		grad := x.Grad
		parallelForWork(b*t*heads, dim, func(start, end int) {
			for idx := start; idx < end; idx++ {
				h := idx % heads
				tmp := idx / heads
				ti := tmp % t
				bi := tmp / t
				dst := ((bi*heads+h)*t + ti) * dim
				src := (bi*t+ti)*model + h*dim
				for d := 0; d < dim; d++ {
					grad[dst+d] += g[src+d]
				}
			}
		})
	})
}

func (m *MultiheadAttention) ForwardSeed(q, k, v, mask *Tensor, seed uint32) *Tensor {
	if inferenceDevice(q.Device) && m.Training {
		panic(gpu.ErrInferenceOnly)
	}
	if len(q.Shape) != 3 || len(k.Shape) != 3 || len(v.Shape) != 3 || q.Shape[0] != k.Shape[0] || k.Shape[0] != v.Shape[0] || k.Shape[1] != v.Shape[1] || q.Shape[2] != m.Model || k.Shape[2] != m.Model || v.Shape[2] != m.Model {
		panic("autograd: attention shape")
	}
	b, t, source := q.Shape[0], q.Shape[1], k.Shape[1]
	headDim := m.Model / m.Heads
	weight, bias := m.param("in_proj_weight"), m.param("in_proj_bias")
	var qh, kh, vh *Tensor
	if q == k && k == v {
		qkv := Add(matMulTransB(q, weight), bias)
		switch q.Device {
		case tensor.CUDA:
			qh = gpuSplitHeads(qkv, b, t, m.Heads, headDim, 3, 0)
			kh = gpuSplitHeads(qkv, b, t, m.Heads, headDim, 3, 1)
			vh = gpuSplitHeads(qkv, b, t, m.Heads, headDim, 3, 2)
		case tensor.CPU:
			qh = cpuSplitHeads(qkv, b, t, m.Heads, headDim, 3, 0)
			kh = cpuSplitHeads(qkv, b, t, m.Heads, headDim, 3, 1)
			vh = cpuSplitHeads(qkv, b, t, m.Heads, headDim, 3, 2)
		default:
			offStep := m.Model
			if b == 0 || t == 0 {
				offStep = 0
			}
			viewShape := []int{b, m.Heads, t, headDim}
			viewStrides := []int{t * 3 * m.Model, headDim, 3 * m.Model, 1}
			qh = makeView(qkv, viewShape, viewStrides, 0)
			kh = makeView(qkv, viewShape, viewStrides, offStep)
			vh = makeView(qkv, viewShape, viewStrides, 2*offStep)
		}
	} else {
		project := func(x *Tensor, start, seq int) *Tensor {
			w := Slice(weight, 0, start, start+m.Model)
			z := Add(matMulTransB(x, w), Slice(bias, 0, start, start+m.Model))
			switch x.Device {
			case tensor.CUDA:
				return gpuSplitHeads(z, b, seq, m.Heads, headDim, 1, 0)
			case tensor.CPU:
				return cpuSplitHeads(z, b, seq, m.Heads, headDim, 1, 0)
			default:
				return Permute(Reshape(z, b, seq, m.Heads, headDim), 0, 2, 1, 3)
			}
		}
		qh, kh, vh = project(q, 0, t), project(k, m.Model, source), project(v, 2*m.Model, source)
	}
	att := attentionWithDropout(qh, kh, vh, mask, m.Dropout, seed, m.Training)
	if len(att.Shape) != 4 || att.Shape[0] != b || att.Shape[1] != m.Heads || att.Shape[2] != t || att.Shape[3] != headDim {
		panic("autograd: reshape size")
	}
	var merged *Tensor
	switch q.Device {
	case tensor.CUDA:
		merged = gpuMergeHeads(att, b, t, m.Heads, headDim)
	case tensor.CPU:
		merged = cpuMergeHeads(att, b, t, m.Heads, headDim)
	default:
		merged = Reshape(Permute(att, 0, 2, 1, 3), b, t, m.Model)
	}
	return Add(matMulTransB(merged, m.param("out_proj.weight")), m.param("out_proj.bias"))
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
	if feedForward < 1 {
		return nil, fmt.Errorf("autograd: invalid encoder feedForward dimension")
	}
	att, e := NewMultiheadAttention(model, heads, device)
	if e != nil {
		return nil, e
	}
	m := &TransformerEncoderLayer{Attention: att, Model: model, FeedForward: feedForward, Training: true}
	m.Module.Training = true
	m.Module.OnTrainingChange = func(training bool) { m.Training = training }
	m.Children = []NamedModule{{"self_attn", &att.Module}}
	cleanup := func() {
		closeParameters(m.Parameters)
		closeParameters(att.Parameters)
	}
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
			cleanup()
			return nil, e
		}
		if p.name == "norm1.weight" || p.name == "norm2.weight" {
			ones := make([]float32, model)
			for j := range ones {
				ones[j] = 1
			}
			if e := v.CopyFrom(ones); e != nil {
				v.Close()
				cleanup()
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
	if inferenceDevice(x.Device) && (m.Training || m.Attention.Training) {
		panic(gpu.ErrInferenceOnly)
	}
	h := LayerNorm(x, m.param("norm1.weight"), m.param("norm1.bias"), 1e-5)
	h = m.Attention.ForwardSeed(h, h, h, mask, stepSeed+3)
	x = Add(x, Dropout(h, m.Dropout, stepSeed, m.Training))
	h = LayerNorm(x, m.param("norm2.weight"), m.param("norm2.bias"), 1e-5)
	h = BiasGELU(matMulTransB(h, m.param("linear1.weight")), m.param("linear1.bias"))
	h = Dropout(h, m.Dropout, stepSeed+1, m.Training)
	h = matMulTransB(h, m.param("linear2.weight"))
	if !m.Training || m.Dropout == 0 {
		return BiasResidual(h, m.param("linear2.bias"), x)
	}
	h = Add(h, m.param("linear2.bias"))
	return Add(x, Dropout(h, m.Dropout, stepSeed+2, true))
}
