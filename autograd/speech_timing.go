package autograd

import (
	"fmt"
	"math/rand"

	"github.com/yh2237/gograd/tensor"
)

var SpeechTimingDilations = []int{1, 2, 4, 8, 1, 2, 4, 8}

// SpeechTiming composes the UtauTTS Target model. Activations use [batch,time,channel].
// Its Module emits the exact PyTorch state_dict names expected by LoadTCN.
type SpeechTiming struct {
	Module        Module
	Phone         *EmbeddingLayer
	Input, Output *Conv1dLayer
	Blocks        []*Conv1dLayer
	Norms         []*LayerNormLayer
	Dropout       DropoutLayer
	GELU          GELULayer
	Continuous    int
}

// NewSpeechTiming builds the UtauTTS Target model with the original 40-phone vocabulary.
func NewSpeechTiming(continuous int, device tensor.Device, seed int64) (*SpeechTiming, error) {
	return NewSpeechTimingWithPhones(40, continuous, device, seed)
}

// NewSpeechTimingWithPhones builds the model for a language-specific phone vocabulary.
// phones must match the embedding size expected by the runtime model metadata.
func NewSpeechTimingWithPhones(phones, continuous int, device tensor.Device, seed int64) (*SpeechTiming, error) {
	if continuous < 1 {
		return nil, fmt.Errorf("autograd: invalid continuous width")
	}
	if phones < 1 {
		return nil, fmt.Errorf("autograd: invalid phone count")
	}
	rng := rand.New(rand.NewSource(seed))
	m := &SpeechTiming{Continuous: continuous, Dropout: DropoutLayer{Probability: .15, Training: true}}
	var e error
	if m.Phone, e = NewEmbeddingLayer(phones, 32, -1, device, rng); e != nil {
		return nil, e
	}
	if m.Input, e = NewConv1dLayer(96+continuous, 128, 1, 1, device, rng); e != nil {
		return nil, e
	}
	m.Module.Children = append(m.Module.Children, NamedModule{"phone", m.Phone.StateModule()}, NamedModule{"inp", m.Input.StateModule()})
	for i, d := range SpeechTimingDilations {
		block, e := NewConv1dLayer(128, 128, 5, d, device, rng)
		if e != nil {
			return nil, e
		}
		norm, e := NewLayerNormLayer(128, 1e-5, device)
		if e != nil {
			return nil, e
		}
		m.Blocks = append(m.Blocks, block)
		m.Norms = append(m.Norms, norm)
		m.Module.Children = append(m.Module.Children, NamedModule{fmt.Sprintf("blocks.%d", i), block.StateModule()}, NamedModule{fmt.Sprintf("norms.%d", i), norm.StateModule()})
	}
	if m.Output, e = NewConv1dLayer(128, 80, 1, 1, device, rng); e != nil {
		return nil, e
	}
	m.Module.Children = append(m.Module.Children, NamedModule{"out", m.Output.StateModule()})
	m.Module.OnTrainingChange = m.Dropout.Train
	m.Module.Train(true)
	return m, nil
}
func (m *SpeechTiming) Parameters() []Parameter { return m.Module.NamedParameters() }
func (m *SpeechTiming) Train(training bool)     { m.Module.Train(training) }
func (m *SpeechTiming) Forward(ids []int, cont *Tensor, seed uint32) *Tensor {
	s := cont.Shape
	if len(s) != 3 || s[2] != m.Continuous || len(ids) != s[0]*s[1]*3 {
		panic("autograd: speech timing input shape")
	}
	phone := Reshape(m.Phone.Forward(ids, []int{s[0], s[1], 3}), s[0], s[1], 96)
	h := m.Input.Forward(Concat(2, phone, cont))
	for i := range m.Blocks {
		y := m.Blocks[i].Forward(h)
		y = m.Norms[i].Forward(y)
		y = m.GELU.Forward(y)
		y = m.Dropout.Forward(y, seed+uint32(i))
		h = Add(h, y)
	}
	return m.Output.Forward(h)
}

// SpeechTimingMultiHead adds a context-only F0 branch to SpeechTiming.
// The F0 branch never receives F0 inputs, so it cannot copy the answer.
type SpeechTimingMultiHead struct {
	Module        Module
	Phone         *EmbeddingLayer
	Input, Output *Conv1dLayer
	Blocks        []*Conv1dLayer
	Norms         []*LayerNormLayer
	Dropout       DropoutLayer
	GELU          GELULayer
	Continuous    int

	F0Input   *Conv1dLayer
	F0Blocks  []*Conv1dLayer
	F0Norms   []*LayerNormLayer
	F0Output  *Conv1dLayer
	F0Context int
}

// NewSpeechTimingMultiHead builds the mel trunk plus a context-only F0 trunk.
// f0Context is the width of the F0-branch continuous input (position and duration).
func NewSpeechTimingMultiHead(phones, continuous, f0Context int, device tensor.Device, seed int64) (*SpeechTimingMultiHead, error) {
	if continuous < 1 {
		return nil, fmt.Errorf("autograd: invalid continuous width")
	}
	if phones < 1 {
		return nil, fmt.Errorf("autograd: invalid phone count")
	}
	if f0Context < 1 {
		return nil, fmt.Errorf("autograd: invalid f0 context width")
	}
	rng := rand.New(rand.NewSource(seed))
	m := &SpeechTimingMultiHead{Continuous: continuous, F0Context: f0Context, Dropout: DropoutLayer{Probability: .15, Training: true}}
	var e error
	if m.Phone, e = NewEmbeddingLayer(phones, 32, -1, device, rng); e != nil {
		return nil, e
	}
	if m.Input, e = NewConv1dLayer(96+continuous, 128, 1, 1, device, rng); e != nil {
		return nil, e
	}
	m.Module.Children = append(m.Module.Children, NamedModule{"phone", m.Phone.StateModule()}, NamedModule{"inp", m.Input.StateModule()})
	for i, d := range SpeechTimingDilations {
		block, e := NewConv1dLayer(128, 128, 5, d, device, rng)
		if e != nil {
			return nil, e
		}
		norm, e := NewLayerNormLayer(128, 1e-5, device)
		if e != nil {
			return nil, e
		}
		m.Blocks = append(m.Blocks, block)
		m.Norms = append(m.Norms, norm)
		m.Module.Children = append(m.Module.Children, NamedModule{fmt.Sprintf("blocks.%d", i), block.StateModule()}, NamedModule{fmt.Sprintf("norms.%d", i), norm.StateModule()})
	}
	if m.Output, e = NewConv1dLayer(128, 80, 1, 1, device, rng); e != nil {
		return nil, e
	}
	m.Module.Children = append(m.Module.Children, NamedModule{"out", m.Output.StateModule()})
	if m.F0Input, e = NewConv1dLayer(96+f0Context, 128, 1, 1, device, rng); e != nil {
		return nil, e
	}
	m.Module.Children = append(m.Module.Children, NamedModule{"f0_inp", m.F0Input.StateModule()})
	for i, d := range SpeechTimingDilations {
		block, e := NewConv1dLayer(128, 128, 5, d, device, rng)
		if e != nil {
			return nil, e
		}
		norm, e := NewLayerNormLayer(128, 1e-5, device)
		if e != nil {
			return nil, e
		}
		m.F0Blocks = append(m.F0Blocks, block)
		m.F0Norms = append(m.F0Norms, norm)
		m.Module.Children = append(m.Module.Children, NamedModule{fmt.Sprintf("f0_blocks.%d", i), block.StateModule()}, NamedModule{fmt.Sprintf("f0_norms.%d", i), norm.StateModule()})
	}
	if m.F0Output, e = NewConv1dLayer(128, 1, 1, 1, device, rng); e != nil {
		return nil, e
	}
	m.Module.Children = append(m.Module.Children, NamedModule{"f0_out", m.F0Output.StateModule()})
	m.Module.OnTrainingChange = m.Dropout.Train
	m.Module.Train(true)
	return m, nil
}

func (m *SpeechTimingMultiHead) Parameters() []Parameter { return m.Module.NamedParameters() }
func (m *SpeechTimingMultiHead) Train(training bool)     { m.Module.Train(training) }

// Forward returns the mel prediction and the context-only F0 prediction.
func (m *SpeechTimingMultiHead) Forward(ids []int, cont, f0Cont *Tensor, seed uint32) (*Tensor, *Tensor) {
	s := cont.Shape
	if len(s) != 3 || s[2] != m.Continuous || len(ids) != s[0]*s[1]*3 {
		panic("autograd: speech timing input shape")
	}
	phone := Reshape(m.Phone.Forward(ids, []int{s[0], s[1], 3}), s[0], s[1], 96)
	h := m.Input.Forward(Concat(2, phone, cont))
	for i := range m.Blocks {
		y := m.Blocks[i].Forward(h)
		y = m.Norms[i].Forward(y)
		y = m.GELU.Forward(y)
		y = m.Dropout.Forward(y, seed+uint32(i))
		h = Add(h, y)
	}
	mel := m.Output.Forward(h)

	f := f0Cont.Shape
	if len(f) != 3 || f[2] != m.F0Context || f[0] != s[0] || f[1] != s[1] {
		panic("autograd: speech timing f0 input shape")
	}
	g := m.F0Input.Forward(Concat(2, phone, f0Cont))
	for i := range m.F0Blocks {
		y := m.F0Blocks[i].Forward(g)
		y = m.F0Norms[i].Forward(y)
		y = m.GELU.Forward(y)
		y = m.Dropout.Forward(y, seed+uint32(64+i))
		g = Add(g, y)
	}
	return mel, m.F0Output.Forward(g)
}
