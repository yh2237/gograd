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

// safeTensorEntryはsafetensorsの1テンソル。
type safeTensorEntry struct {
	Values []float32
	Shape  []int
}

// writeSafetensorsはF32テンソル群をsafetensors v1形式で書く。
func writeSafetensors(path string, entries map[string]safeTensorEntry, metadata map[string]string) error {
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	header := make(map[string]safeHeader, len(names))
	var raw bytes.Buffer
	for _, name := range names {
		entry := entries[name]
		start := raw.Len()
		for _, v := range entry.Values {
			if err := encodingbinary.Write(&raw, encodingbinary.LittleEndian, math.Float32bits(v)); err != nil {
				return err
			}
		}
		shape := make([]int, len(entry.Shape))
		copy(shape, entry.Shape)
		header[name] = safeHeader{"F32", shape, [2]int{start, raw.Len()}}
	}
	jsonEntries := make(map[string]any, len(header)+1)
	for name, entry := range header {
		jsonEntries[name] = entry
	}
	if metadata != nil {
		jsonEntries["__metadata__"] = metadata
	}
	h, err := json.Marshal(jsonEntries)
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

// readSafetensorsは全テンソルとmetadataを返す。
func readSafetensors(path string) (map[string]safeTensorEntry, map[string]string, error) {
	file, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	if len(file) < 8 {
		return nil, nil, fmt.Errorf("autograd: short safetensors file")
	}
	hlen := encodingbinary.LittleEndian.Uint64(file[:8])
	if hlen > uint64(len(file)-8) {
		return nil, nil, fmt.Errorf("autograd: invalid safetensors header length")
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(file[8:8+int(hlen)], &header); err != nil {
		return nil, nil, err
	}
	metadata := map[string]string{}
	if rawMetadata, ok := header["__metadata__"]; ok {
		if err := json.Unmarshal(rawMetadata, &metadata); err != nil {
			return nil, nil, fmt.Errorf("autograd: invalid safetensors metadata: %w", err)
		}
		delete(header, "__metadata__")
	}
	data := file[8+int(hlen):]
	entries := make(map[string]safeTensorEntry, len(header))
	for name, rawEntry := range header {
		var meta safeHeader
		if err := json.Unmarshal(rawEntry, &meta); err != nil {
			return nil, nil, fmt.Errorf("autograd: %s: %w", name, err)
		}
		if meta.DType != "F32" {
			return nil, nil, fmt.Errorf("autograd: incompatible dtype %s for %s", meta.DType, name)
		}
		count := 1
		for _, d := range meta.Shape {
			if d < 0 {
				return nil, nil, fmt.Errorf("autograd: invalid shape %s", name)
			}
			count *= d
		}
		start, end := meta.Offsets[0], meta.Offsets[1]
		if start < 0 || end < start || end > len(data) || end-start != count*4 {
			return nil, nil, fmt.Errorf("autograd: invalid offsets %s", name)
		}
		values := make([]float32, count)
		for i := range values {
			values[i] = math.Float32frombits(encodingbinary.LittleEndian.Uint32(data[start+i*4:]))
		}
		entries[name] = safeTensorEntry{Values: values, Shape: meta.Shape}
	}
	return entries, metadata, nil
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
	entries := make(map[string]safeTensorEntry, len(state))
	for name, t := range state {
		if t.DType != Float32 {
			return fmt.Errorf("autograd: unsupported dtype %s", t.DType)
		}
		values, err := t.ToHost()
		if err != nil {
			return err
		}
		shape := make([]int, len(t.Shape))
		copy(shape, t.Shape)
		entries[name] = safeTensorEntry{Values: values, Shape: shape}
	}
	return writeSafetensors(path, entries, metadata)
}

// LoadSafeTensors validates all names, shapes, dtype, and byte ranges before
// copying any weights into the module.
func (m *Module) LoadSafeTensors(path string) error {
	entries, _, err := readSafetensors(path)
	if err != nil {
		return err
	}
	state := m.namedState()
	if len(entries) != len(state) {
		return fmt.Errorf("autograd: safetensors entry count %d != %d", len(entries), len(state))
	}
	for name, t := range state {
		entry, ok := entries[name]
		if !ok {
			return fmt.Errorf("autograd: missing tensor %s", name)
		}
		if len(entry.Shape) != len(t.Shape) {
			return fmt.Errorf("autograd: incompatible tensor %s", name)
		}
		for i, d := range t.Shape {
			if d != entry.Shape[i] {
				return fmt.Errorf("autograd: shape mismatch %s", name)
			}
		}
		if len(entry.Values) != t.Numel() {
			return fmt.Errorf("autograd: invalid value count %s", name)
		}
	}
	for name, t := range state {
		if err := t.CopyFrom(entries[name].Values); err != nil {
			return err
		}
	}
	return nil
}
