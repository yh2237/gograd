package autograd

import (
	"bytes"
	encodingbinary "encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

type safeHeader struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

func (m *Module) namedState() map[string]*Tensor {
	out := map[string]*Tensor{}
	var walk func(string, *Module)
	walk = func(prefix string, node *Module) {
		for _, p := range node.Parameters {
			out[prefix+p.Name] = p.Value
		}
		for _, p := range node.Buffers {
			out[prefix+p.Name] = p.Value
		}
		for _, c := range node.Children {
			walk(prefix+c.Name+".", c.Module)
		}
	}
	walk("", m)
	return out
}

// SaveSafeTensors writes F32 tensors using the safetensors v1 file layout.
func (m *Module) SaveSafeTensors(path string) error {
	return m.SaveSafeTensorsMetadata(path, nil)
}

// SaveSafeTensorsMetadata writes state_dict weights and string metadata in a
// format accepted by PyTorch safetensors and UtauTTS's speech-timing loader.
func (m *Module) SaveSafeTensorsMetadata(path string, metadata map[string]string) error {
	state := m.namedState()
	names := make([]string, 0, len(state))
	for name := range state {
		names = append(names, name)
	}
	sort.Strings(names)
	header := make(map[string]safeHeader, len(names))
	var raw bytes.Buffer
	for _, name := range names {
		t := state[name]
		if t.DType != Float32 {
			return fmt.Errorf("autograd: unsupported dtype %s", t.DType)
		}
		values, err := t.ToHost()
		if err != nil {
			return err
		}
		start := raw.Len()
		for _, v := range values {
			if err := encodingbinary.Write(&raw, encodingbinary.LittleEndian, math.Float32bits(v)); err != nil {
				return err
			}
		}
		shape := make([]int, len(t.Shape))
		copy(shape, t.Shape)
		header[name] = safeHeader{"F32", shape, [2]int{start, raw.Len()}}
	}
	entries := make(map[string]any, len(header)+1)
	for name, entry := range header {
		entries[name] = entry
	}
	if metadata != nil {
		entries["__metadata__"] = metadata
	}
	h, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	var file bytes.Buffer
	if err := encodingbinary.Write(&file, encodingbinary.LittleEndian, uint64(len(h))); err != nil {
		return err
	}
	file.Write(h)
	file.Write(raw.Bytes())
	return os.WriteFile(path, file.Bytes(), 0600)
}

// LoadSafeTensors validates all names, shapes, dtype, and byte ranges before
// copying any weights into the module.
func (m *Module) LoadSafeTensors(path string) error {
	file, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(file) < 8 {
		return fmt.Errorf("autograd: short safetensors file")
	}
	hlen := encodingbinary.LittleEndian.Uint64(file[:8])
	if hlen > uint64(len(file)-8) {
		return fmt.Errorf("autograd: invalid safetensors header length")
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(file[8:8+int(hlen)], &header); err != nil {
		return err
	}
	delete(header, "__metadata__")
	state := m.namedState()
	if len(header) != len(state) {
		return fmt.Errorf("autograd: safetensors entry count %d != %d", len(header), len(state))
	}
	data := file[8+int(hlen):]
	values := make(map[string][]float32, len(state))
	for name, t := range state {
		entry, ok := header[name]
		if !ok {
			return fmt.Errorf("autograd: missing tensor %s", name)
		}
		var meta safeHeader
		if err := json.Unmarshal(entry, &meta); err != nil {
			return fmt.Errorf("autograd: %s: %w", name, err)
		}
		if meta.DType != "F32" || len(meta.Shape) != len(t.Shape) {
			return fmt.Errorf("autograd: incompatible tensor %s", name)
		}
		for i, d := range t.Shape {
			if d != meta.Shape[i] {
				return fmt.Errorf("autograd: shape mismatch %s", name)
			}
		}
		start, end := meta.Offsets[0], meta.Offsets[1]
		if start < 0 || end < start || end > len(data) || end-start != t.Numel()*4 {
			return fmt.Errorf("autograd: invalid offsets %s", name)
		}
		v := make([]float32, t.Numel())
		for i := range v {
			v[i] = math.Float32frombits(encodingbinary.LittleEndian.Uint32(data[start+i*4:]))
		}
		values[name] = v
	}
	for name, t := range state {
		if err := t.CopyFrom(values[name]); err != nil {
			return err
		}
	}
	return nil
}
