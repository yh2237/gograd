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
