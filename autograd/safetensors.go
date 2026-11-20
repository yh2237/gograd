package autograd

import (
	encodingbinary "encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
)

const (
	safeTensorHeaderLimit = 16 << 20
	safeTensorChunkValues = 16 << 10
)

type safeHeader struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

type safeTensorEntry struct {
	Values []float32
	Shape  []int
}

func writeSafetensors(path string, entries map[string]safeTensorEntry, metadata map[string]string) error {
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	header := make(map[string]any, len(names)+1)
	offset := 0
	for _, name := range names {
		entry := entries[name]
		count, err := safeNumel(entry.Shape, len(entry.Values))
		if err != nil || count != len(entry.Values) || name == "__metadata__" {
			return fmt.Errorf("autograd: incompatible tensor %s", name)
		}
		if count > (math.MaxInt-offset)/4 {
			return fmt.Errorf("autograd: oversized tensor data")
		}
		end := offset + count*4
		header[name] = safeHeader{"F32", append([]int{}, entry.Shape...), [2]int{offset, end}}
		offset = end
	}
	if metadata != nil {
		header["__metadata__"] = metadata
	}
	h, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(h)%8 != 0 {
		h = append(h, ' ')
	}
	if len(h) > safeTensorHeaderLimit {
		return fmt.Errorf("autograd: invalid safetensors header length")
	}

	file, err := os.CreateTemp(filepath.Dir(path), ".gograd-checkpoint-*")
	if err != nil {
		return err
	}
	name := file.Name()
	defer os.Remove(name)
	defer file.Close()
	var prefix [8]byte
	encodingbinary.LittleEndian.PutUint64(prefix[:], uint64(len(h)))
	if _, err := file.Write(prefix[:]); err != nil {
		return err
	}
	if _, err := file.Write(h); err != nil {
		return err
	}
	var scratch [safeTensorChunkValues * 4]byte
	for _, name := range names {
		for chunk := range slices.Chunk(entries[name].Values, safeTensorChunkValues) {
			for i, v := range chunk {
				encodingbinary.LittleEndian.PutUint32(scratch[i*4:], math.Float32bits(v))
			}
			if _, err := file.Write(scratch[:len(chunk)*4]); err != nil {
				return err
			}
		}
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}

func safeNumel(shape []int, limit int) (int, error) {
	zero := false
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("negative dimension")
		}
		zero = zero || d == 0
	}
	if zero {
		return 0, nil
	}
	count := 1
	for _, d := range shape {
		if d > limit || count > limit/d {
			return 0, fmt.Errorf("oversized shape")
		}
		count *= d
	}
	if count > limit {
		return 0, fmt.Errorf("oversized shape")
	}
	return count, nil
}

func readSafetensors(path string) (map[string]safeTensorEntry, map[string]string, error) {
	file, err := OpenSafeTensorFile(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	entries := make(map[string]safeTensorEntry, len(file.entries))
	for _, name := range file.Names() {
		values, shape, err := file.ReadF32(name)
		if err != nil {
			return nil, nil, err
		}
		entries[name] = safeTensorEntry{Values: values, Shape: shape}
	}
	return entries, file.Metadata, nil
}

// namedState flattens the module tree into dotted names. Parameters and buffers
// share one namespace, so a collision is reported instead of silently dropping
// a tensor from the checkpoint.
func (m *Module) namedState() (map[string]*Tensor, error) {
	out := map[string]*Tensor{}
	var walk func(string, *Module) error
	walk = func(prefix string, node *Module) error {
		add := func(name string, value *Tensor) error {
			key := prefix + name
			if _, ok := out[key]; ok {
				return fmt.Errorf("autograd: duplicate state name %s", key)
			}
			out[key] = value
			return nil
		}
		for _, p := range node.Parameters {
			if err := add(p.Name, p.Value); err != nil {
				return err
			}
		}
		for _, p := range node.Buffers {
			if err := add(p.Name, p.Value); err != nil {
				return err
			}
		}
		for _, c := range node.Children {
			if err := walk(prefix+c.Name+".", c.Module); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk("", m); err != nil {
		return nil, err
	}
	return out, nil
}

// SaveSafeTensors writes F32 tensors using the safetensors v1 file layout.
func (m *Module) SaveSafeTensors(path string) error {
	return m.SaveSafeTensorsMetadata(path, nil)
}

// SaveSafeTensorsMetadata writes state_dict weights and string metadata in a
// format accepted by PyTorch safetensors and UtauTTS's speech-timing loader.
func (m *Module) SaveSafeTensorsMetadata(path string, metadata map[string]string) error {
	entries, err := m.stateEntries()
	if err != nil {
		return err
	}
	return writeSafetensors(path, entries, metadata)
}

func (m *Module) stateEntries() (map[string]safeTensorEntry, error) {
	state, err := m.namedState()
	if err != nil {
		return nil, err
	}
	entries := make(map[string]safeTensorEntry, len(state))
	for name, t := range state {
		if t.DType != Float32 {
			return nil, fmt.Errorf("autograd: unsupported dtype %s", t.DType)
		}
		values, err := t.ToHost()
		if err != nil {
			return nil, err
		}
		shape := make([]int, len(t.Shape))
		copy(shape, t.Shape)
		entries[name] = safeTensorEntry{Values: values, Shape: shape}
	}
	return entries, nil
}

// LoadSafeTensors validates all names, shapes, dtype, and byte ranges before
// copying any weights into the module.
func (m *Module) LoadSafeTensors(path string) error {
	entries, _, err := readSafetensors(path)
	if err != nil {
		return err
	}
	if err := m.validateEntries(entries); err != nil {
		return err
	}
	return m.loadEntries(entries)
}

func (m *Module) validateEntries(entries map[string]safeTensorEntry) error {
	state, err := m.namedState()
	if err != nil {
		return err
	}
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
		if !slices.Equal(t.Shape, entry.Shape) {
			return fmt.Errorf("autograd: shape mismatch %s", name)
		}
		if len(entry.Values) != t.Numel() {
			return fmt.Errorf("autograd: invalid value count %s", name)
		}
	}
	return nil
}

func (m *Module) loadEntries(entries map[string]safeTensorEntry) error {
	state, err := m.namedState()
	if err != nil {
		return err
	}
	for name, t := range state {
		if err := t.CopyFrom(entries[name].Values); err != nil {
			return err
		}
	}
	return nil
}
