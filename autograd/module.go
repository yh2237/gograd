package autograd

import "fmt"

// Trainerは学習モードを切り替えられるモジュール。
type Trainer interface{ Train(training bool) }

// SetTrainingは学習モードをまとめて切り替える。
func SetTraining(training bool, trainers ...Trainer) {
	for _, trainer := range trainers {
		if trainer != nil {
			trainer.Train(training)
		}
	}
}

// Module is an ordered parameter/buffer registry. StateDict returns copies so
// checkpoint writers cannot mutate live tensors.
type Module struct {
	Parameters []Parameter
	Buffers    []Parameter
	Children   []NamedModule
}
type NamedModule struct {
	Name   string
	Module *Module
}

func (m *Module) NamedParameters() []Parameter {
	var out []Parameter
	var walk func(string, *Module)
	walk = func(prefix string, node *Module) {
		for _, p := range node.Parameters {
			out = append(out, Parameter{prefix + p.Name, p.Value})
		}
		for _, child := range node.Children {
			walk(prefix+child.Name+".", child.Module)
		}
	}
	walk("", m)
	return out
}
func (m *Module) StateDict() map[string][]float32 {
	out := map[string][]float32{}
	var walk func(string, *Module)
	walk = func(prefix string, node *Module) {
		for _, p := range append(append([]Parameter(nil), node.Parameters...), node.Buffers...) {
			values, e := p.Value.ToHost()
			if e != nil {
				panic(e)
			}
			out[prefix+p.Name] = values
		}
		for _, child := range node.Children {
			walk(prefix+child.Name+".", child.Module)
		}
	}
	walk("", m)
	return out
}
func (m *Module) LoadStateDict(state map[string][]float32) error {
	all := m.StateDict()
	if len(all) != len(state) {
		return fmt.Errorf("autograd: state entry count %d != %d", len(state), len(all))
	}
	for name, current := range all {
		v, ok := state[name]
		if !ok || len(v) != len(current) {
			return fmt.Errorf("autograd: missing or mismatched %s", name)
		}
	}
	var walk func(string, *Module) error
	walk = func(prefix string, node *Module) error {
		for _, p := range append(append([]Parameter(nil), node.Parameters...), node.Buffers...) {
			if e := p.Value.CopyFrom(state[prefix+p.Name]); e != nil {
				return e
			}
		}
		for _, child := range node.Children {
			if e := walk(prefix+child.Name+".", child.Module); e != nil {
				return e
			}
		}
		return nil
	}
	return walk("", m)
}
