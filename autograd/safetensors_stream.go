package autograd

import (
	encodingbinary "encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sort"
	"sync"
)

// SafeTensorFile reads individual F32 tensors without retaining the full
// checkpoint in host memory. This is suitable for multi-GB HF checkpoints.
type SafeTensorFile struct {
	file     *os.File
	dataBase int64
	entries  map[string]safeHeader
	Metadata map[string]string
	cacheMu  sync.RWMutex
	cache    map[string][]float32
}

// OpenSafeTensorFile validates the complete safetensors index before exposing
// any tensor. The payload is read only when ReadF32 is called.
func OpenSafeTensorFile(path string) (*SafeTensorFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	closeOnError := func(err error) (*SafeTensorFile, error) { f.Close(); return nil, err }
	stat, err := f.Stat()
	if err != nil {
		return closeOnError(err)
	}
	if stat.Size() < 8 {
		return closeOnError(fmt.Errorf("autograd: short safetensors file"))
	}
	var prefix [8]byte
	if _, err := io.ReadFull(f, prefix[:]); err != nil {
		return closeOnError(err)
	}
	hlen := encodingbinary.LittleEndian.Uint64(prefix[:])
	if hlen > 16<<20 || hlen > uint64(stat.Size()-8) {
		return closeOnError(fmt.Errorf("autograd: invalid safetensors header length"))
	}
	headerBytes := make([]byte, int(hlen))
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return closeOnError(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return closeOnError(err)
	}
	metadata := map[string]string{}
	if value, ok := raw["__metadata__"]; ok {
		if err := json.Unmarshal(value, &metadata); err != nil {
			return closeOnError(err)
		}
		delete(raw, "__metadata__")
	}
	entries := make(map[string]safeHeader, len(raw))
	var ranges [][2]int
	dataSize := stat.Size() - 8 - int64(hlen)
	for name, value := range raw {
		var h safeHeader
		if err := json.Unmarshal(value, &h); err != nil {
			return closeOnError(fmt.Errorf("autograd: %s: %w", name, err))
		}
		if h.DType != "F32" {
			return closeOnError(fmt.Errorf("autograd: incompatible dtype %s for %s", h.DType, name))
		}
		if dataSize/4 > int64(int(^uint(0)>>1)) {
			return closeOnError(fmt.Errorf("autograd: oversized tensor data"))
		}
		count, err := safeNumel(h.Shape, int(dataSize/4))
		if err != nil {
			return closeOnError(fmt.Errorf("autograd: %s: %w", name, err))
		}
		start, end := h.Offsets[0], h.Offsets[1]
		if start < 0 || end < start || int64(end) > dataSize || end-start != count*4 {
			return closeOnError(fmt.Errorf("autograd: invalid offsets %s", name))
		}
		if end > start {
			ranges = append(ranges, h.Offsets)
		}
		entries[name] = h
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i][0] < ranges[j][0] })
	for i := 1; i < len(ranges); i++ {
		if ranges[i][0] < ranges[i-1][1] {
			return closeOnError(fmt.Errorf("autograd: overlapping tensor ranges"))
		}
	}
	return &SafeTensorFile{file: f, dataBase: 8 + int64(hlen), entries: entries, Metadata: metadata}, nil
}

func (s *SafeTensorFile) Close() error { return s.file.Close() }

// Names returns sorted tensor names in the checkpoint.
func (s *SafeTensorFile) Names() []string {
	names := make([]string, 0, len(s.entries))
	for name := range s.entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (s *SafeTensorFile) Shape(name string) ([]int, error) {
	h, ok := s.entries[name]
	if !ok {
		return nil, fmt.Errorf("autograd: missing tensor %s", name)
	}
	return slices.Clone(h.Shape), nil
}

// ReadF32 reads only the named tensor and returns its shape and values.
func (s *SafeTensorFile) ReadF32(name string) ([]float32, []int, error) {
	h, ok := s.entries[name]
	if !ok {
		return nil, nil, fmt.Errorf("autograd: missing tensor %s", name)
	}
	s.cacheMu.RLock()
	if values, ok := s.cache[name]; ok {
		s.cacheMu.RUnlock()
		return values, slices.Clone(h.Shape), nil
	}
	s.cacheMu.RUnlock()
	n := (h.Offsets[1] - h.Offsets[0]) / 4
	raw := make([]byte, n*4)
	if _, err := s.file.ReadAt(raw, s.dataBase+int64(h.Offsets[0])); err != nil {
		return nil, nil, err
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = math.Float32frombits(encodingbinary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values, slices.Clone(h.Shape), nil
}

// PreloadF32 keeps selected checkpoint tensors resident in host memory.
// Cached slices returned by ReadF32 are shared and must be treated as read-only.
// Passing no names preloads the entire checkpoint. Call before concurrent reads.
func (s *SafeTensorFile) PreloadF32(names ...string) error {
	if len(names) == 0 {
		names = s.Names()
	}
	for _, name := range names {
		values, _, err := s.ReadF32(name)
		if err != nil {
			return err
		}
		s.cacheMu.Lock()
		if s.cache == nil {
			s.cache = make(map[string][]float32, len(names))
		}
		if _, ok := s.cache[name]; !ok {
			s.cache[name] = values
		}
		s.cacheMu.Unlock()
	}
	return nil
}

// CachedBytes reports the host memory retained by PreloadF32.
func (s *SafeTensorFile) CachedBytes() int64 {
	s.cacheMu.RLock()
	defer s.cacheMu.RUnlock()
	var bytes int64
	for _, v := range s.cache {
		bytes += int64(len(v)) * 4
	}
	return bytes
}

// LoadSafeTensorsStream loads an exact F32 state dict one tensor at a time.
// All names and shapes are checked before any live tensor is changed.
func (m *Module) LoadSafeTensorsStream(path string) error {
	s, err := OpenSafeTensorFile(path)
	if err != nil {
		return err
	}
	defer s.Close()
	state := m.namedState()
	if len(state) != len(s.entries) {
		return fmt.Errorf("autograd: safetensors entry count %d != %d", len(s.entries), len(state))
	}
	for name, t := range state {
		h, ok := s.entries[name]
		if !ok {
			return fmt.Errorf("autograd: missing tensor %s", name)
		}
		if t.DType != Float32 || !slices.Equal(t.Shape, h.Shape) {
			return fmt.Errorf("autograd: incompatible tensor %s", name)
		}
	}
	for _, name := range s.Names() {
		values, _, err := s.ReadF32(name)
		if err != nil {
			return err
		}
		if err := state[name].CopyFrom(values); err != nil {
			return err
		}
	}
	return nil
}
