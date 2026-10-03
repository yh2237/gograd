package autograd

import "fmt"

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
			out[prefix+p.Name] = append([]float32(nil), p.Value.Data...)
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
	var walk func(string, *Module)
	walk = func(prefix string, node *Module) {
		for _, p := range append(append([]Parameter(nil), node.Parameters...), node.Buffers...) {
			copy(p.Value.Data, state[prefix+p.Name])
		}
		for _, child := range node.Children {
			walk(prefix+child.Name+".", child.Module)
		}
	}
	walk("", m)
	return nil
}
