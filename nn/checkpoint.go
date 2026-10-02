package nn

import (
	"encoding/json"
	"fmt"
	"github.com/yh2237/gograd/tensor"
	"io"
)

// LayerConfig describes a layer for an independent inference reader. Layers
// appear in forward order; a residual layer's children live in Layers.
type LayerConfig struct {
	Name        string        `json:"name"`
	Type        string        `json:"type"`
	InChannels  int           `json:"in_channels,omitempty"`
	OutChannels int           `json:"out_channels,omitempty"`
	Kernel      int           `json:"kernel,omitempty"`
	Dilation    int           `json:"dilation,omitempty"`
	Activation  string        `json:"activation,omitempty"`
	Slope       float32       `json:"slope,omitempty"`
	Layers      []LayerConfig `json:"layers,omitempty"`
}
type ParameterJSON struct {
	Name   string    `json:"name"`
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}
type Checkpoint struct {
	Version    int             `json:"version"`
	Layers     []LayerConfig   `json:"layers"`
	Parameters []ParameterJSON `json:"parameters"`
}

// Save writes the model configuration and named float32 parameters as JSON.
func Save(w io.Writer, model Module) error {
	c := Checkpoint{Version: 1}
	var err error
	c.Layers, c.Parameters, err = describe(model, "model")
	if err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(c)
}
func describe(m Module, name string) ([]LayerConfig, []ParameterJSON, error) {
	if s, ok := m.(*Sequential); ok {
		var cfg []LayerConfig
		var params []ParameterJSON
		for i, child := range s.Modules {
			a, b, err := describe(child, fmt.Sprintf("%s.%d", name, i))
			if err != nil {
				return nil, nil, err
			}
			cfg = append(cfg, a...)
			params = append(params, b...)
		}
		return cfg, params, nil
	}
	if r, ok := m.(*Residual); ok {
		children, params, err := describe(r.Inner, name+".inner")
		return []LayerConfig{{Name: name, Type: "residual", Layers: children}}, params, err
	}
	config := LayerConfig{Name: name}
	switch v := m.(type) {
	case *Linear:
		config.Type = "linear"
		config.InChannels = v.in
		config.OutChannels = v.out
	case *Conv1d:
		config.Type = "conv1d"
		config.InChannels = v.In
		config.OutChannels = v.Out
		config.Kernel = v.Kernel
		config.Dilation = v.Dilation
	case *ReLU:
		config.Type = "activation"
		config.Activation = "relu"
	case *LeakyReLU:
		config.Type = "activation"
		config.Activation = "leaky_relu"
		config.Slope = v.Slope
	case *GELU:
		config.Type = "activation"
		config.Activation = "gelu"
	case *Tanh:
		config.Type = "activation"
		config.Activation = "tanh"
	default:
		return nil, nil, fmt.Errorf("nn: cannot save module %T", m)
	}
	params := []ParameterJSON{}
	for i, p := range m.Parameters() {
		suffix := "weight"
		if i == 1 {
			suffix = "bias"
		}
		data, err := p.ToHost()
		if err != nil {
			return nil, nil, err
		}
		params = append(params, ParameterJSON{Name: name + "." + suffix, Shape: p.Shape(), Values: data})
	}
	return []LayerConfig{config}, params, nil
}

// Load reads a checkpoint into an existing model, checking names and shapes.
func Load(r io.Reader, model Module) (*Checkpoint, error) {
	var c Checkpoint
	if err := json.NewDecoder(r).Decode(&c); err != nil {
		return nil, err
	}
	if c.Version != 1 {
		return nil, fmt.Errorf("nn: unsupported checkpoint version %d", c.Version)
	}
	_, expected, err := describe(model, "model")
	if err != nil {
		return nil, err
	}
	if len(expected) != len(c.Parameters) {
		return nil, fmt.Errorf("nn: parameter count mismatch")
	}
	params := model.Parameters()
	for i, p := range params {
		got, want := c.Parameters[i], expected[i]
		if got.Name != want.Name || len(got.Shape) != len(want.Shape) || len(got.Values) != p.Numel() {
			return nil, fmt.Errorf("nn: parameter %d name or shape mismatch", i)
		}
		for j := range got.Shape {
			if got.Shape[j] != want.Shape[j] {
				return nil, fmt.Errorf("nn: parameter %s shape mismatch", got.Name)
			}
		}
		if p.Device() != tensor.CPU && p.Device() != tensor.CUDA {
			return nil, fmt.Errorf("nn: invalid parameter device")
		}
		if err := p.CopyFrom(got.Values); err != nil {
			return nil, err
		}
	}
	return &c, nil
}
