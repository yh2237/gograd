package autograd

import (
	"fmt"
	"github.com/yh2237/gograd/tensor"
)

// Acoustic is the frame acoustic model from the UtauTTS training reference.
// Convolution tensors use [batch,time,channels] internally.
type Acoustic struct {
	Params                            []Parameter
	Hidden, Speakers, Outputs, Blocks int
	Dilations                         []int
	Device                            tensor.Device
}

func NewAcoustic(hidden, speakers, outputs int, dilations []int, device tensor.Device) (*Acoustic, error) {
	m := &Acoustic{Hidden: hidden, Speakers: speakers, Outputs: outputs, Blocks: len(dilations), Dilations: append([]int(nil), dilations...), Device: device}
	add := func(name string, shape ...int) error {
		v, e := Zeros(shape, device, true)
		if e != nil {
			return e
		}
		m.Params = append(m.Params, Parameter{name, v})
		return nil
	}
	for _, p := range []struct {
		name  string
		shape []int
	}{{"phone.weight", []int{46, 64}}, {"speaker.weight", []int{speakers, 64}}, {"inp.weight", []int{hidden, 260, 1}}, {"inp.bias", []int{hidden}}} {
		if e := add(p.name, p.shape...); e != nil {
			return nil, e
		}
	}
	for i, d := range dilations {
		for _, p := range []struct {
			name  string
			shape []int
		}{{fmt.Sprintf("blocks.%d.weight", i), []int{hidden, hidden, 5}}, {fmt.Sprintf("blocks.%d.bias", i), []int{hidden}}, {fmt.Sprintf("norms.%d.weight", i), []int{hidden}}, {fmt.Sprintf("norms.%d.bias", i), []int{hidden}}, {fmt.Sprintf("film.%d.weight", i), []int{2 * hidden, 64}}, {fmt.Sprintf("film.%d.bias", i), []int{2 * hidden}}} {
			if e := add(p.name, p.shape...); e != nil {
				return nil, e
			}
		}
		_ = d
	}
	for _, p := range []struct {
		name  string
		shape []int
	}{{"out.weight", []int{outputs, hidden, 1}}, {"out.bias", []int{outputs}}} {
		if e := add(p.name, p.shape...); e != nil {
			return nil, e
		}
	}
	return m, nil
}
func (m *Acoustic) Param(name string) *Tensor {
	for _, p := range m.Params {
		if p.Name == name {
			return p.Value
		}
	}
	panic("autograd: unknown parameter " + name)
}
func (m *Acoustic) Forward(ids []int, cont *Tensor, speaker []int) *Tensor {
	s := cont.Shape
	if len(s) != 3 || s[2] != 4 || len(ids) != s[0]*s[1]*3 || len(speaker) != s[0] {
		panic("autograd: acoustic inputs")
	}
	b, t := s[0], s[1]
	phone := Reshape(Embedding(m.Param("phone.weight"), ids, []int{b, t, 3}), b, t, 192)
	spk := Embedding(m.Param("speaker.weight"), speaker, []int{b})
	expanded := Add(Reshape(spk, b, 1, 64), mustZeros([]int{b, t, 64}, m.Device))
	x := Concat(2, phone, expanded, cont)
	h := Conv1dGEMM(x, m.Param("inp.weight"), m.Param("inp.bias"), 1)
	for i, d := range m.Dilations {
		tag := func(s string) string { return fmt.Sprintf("%s.%d", s, i) }
		film := Add(MatMul(spk, Transpose(m.Param(tag("film")+".weight"), 0, 1)), m.Param(tag("film")+".bias"))
		scale := Reshape(Slice(film, 1, 0, m.Hidden), b, 1, m.Hidden)
		shift := Reshape(Slice(film, 1, m.Hidden, 2*m.Hidden), b, 1, m.Hidden)
		z := Conv1dGEMM(h, m.Param(tag("blocks")+".weight"), m.Param(tag("blocks")+".bias"), d)
		z = GroupNorm(z, m.Param(tag("norms")+".weight"), m.Param(tag("norms")+".bias"), 1, 1e-5)
		z = GELU(Add(Mul(z, AddScalar(scale, 1)), shift), false)
		h = Add(h, z)
	}
	return Conv1dGEMM(h, m.Param("out.weight"), m.Param("out.bias"), 1)
}
func mustZeros(shape []int, device tensor.Device) *Tensor {
	v, e := Zeros(shape, device, false)
	if e != nil {
		panic(e)
	}
	return v
}
