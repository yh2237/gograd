package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
)

type utterance struct {
	ID           string
	Frames       int
	Continuous   int
	IDs          []int
	Cont, Target []float32
}
type tensorEntry struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

func readFeatures(path string) ([]utterance, error) {
	file, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	if len(file) < 8 {
		return nil, fmt.Errorf("feature cache: short header")
	}
	hlen := binary.LittleEndian.Uint64(file[:8])
	if hlen > uint64(len(file)-8) {
		return nil, fmt.Errorf("feature cache: invalid header length")
	}
	var header map[string]json.RawMessage
	if e = json.Unmarshal(file[8:8+hlen], &header); e != nil {
		return nil, e
	}
	var meta map[string]string
	if e = json.Unmarshal(header["__metadata__"], &meta); e != nil {
		return nil, e
	}
	if meta["format"] != "gograd-speech-timing-features-1" {
		return nil, fmt.Errorf("feature cache: unknown format")
	}
	var names []string
	if e = json.Unmarshal([]byte(meta["ids"]), &names); e != nil {
		return nil, e
	}
	data := file[8+hlen:]
	read := func(name, kind string) ([]int, []float32, []int, error) {
		var entry tensorEntry
		if e := json.Unmarshal(header[name], &entry); e != nil {
			return nil, nil, nil, fmt.Errorf("feature cache %s: %w", name, e)
		}
		if entry.DType != kind || len(entry.Shape) != 2 {
			return nil, nil, nil, fmt.Errorf("feature cache %s: dtype/shape", name)
		}
		count := entry.Shape[0] * entry.Shape[1]
		width := 4
		if kind == "I64" {
			width = 8
		}
		start, end := entry.Offsets[0], entry.Offsets[1]
		if count < 0 || start < 0 || end < start || end > len(data) || end-start != count*width {
			return nil, nil, nil, fmt.Errorf("feature cache %s: offsets", name)
		}
		if kind == "I64" {
			v := make([]int, count)
			for i := range v {
				raw := int64(binary.LittleEndian.Uint64(data[start+i*8:]))
				if raw < 0 || raw >= 40 {
					return nil, nil, nil, fmt.Errorf("feature cache %s: phone id %d", name, raw)
				}
				v[i] = int(raw)
			}
			return v, nil, entry.Shape, nil
		}
		v := make([]float32, count)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[start+i*4:]))
		}
		return nil, v, entry.Shape, nil
	}
	items := make([]utterance, len(names))
	for i, name := range names {
		prefix := fmt.Sprintf("%05d.", i)
		ids, _, is, e := read(prefix+"ids", "I64")
		if e != nil {
			return nil, e
		}
		_, cont, cs, e := read(prefix+"cont", "F32")
		if e != nil {
			return nil, e
		}
		_, target, ts, e := read(prefix+"target", "F32")
		if e != nil {
			return nil, e
		}
		if is[1] != 3 || ts[1] != 80 || cs[1] < 4 || is[0] != cs[0] || is[0] != ts[0] {
			return nil, fmt.Errorf("feature cache %s: incompatible shapes", name)
		}
		items[i] = utterance{name, is[0], cs[1], ids, cont, target}
	}
	return items, nil
}

// readFeatureSplit reads the Python Random shuffle recorded by the exporter.
// A different requested seed falls back to the Go sampler's own split.
func readFeatureSplit(path string, seed int64, count int) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var prefix [8]byte
	if _, err = io.ReadFull(f, prefix[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n > 16<<20 {
		return nil, fmt.Errorf("feature cache: oversized header")
	}
	h := make([]byte, int(n))
	if _, err = io.ReadFull(f, h); err != nil {
		return nil, err
	}
	var header map[string]json.RawMessage
	if err = json.Unmarshal(h, &header); err != nil {
		return nil, err
	}
	var meta map[string]string
	if err = json.Unmarshal(header["__metadata__"], &meta); err != nil {
		return nil, err
	}
	stored, err := strconv.ParseInt(meta["split_seed"], 10, 64)
	if err != nil || stored != seed {
		return nil, nil
	}
	var order []int
	if err = json.Unmarshal([]byte(meta["split_order"]), &order); err != nil {
		return nil, err
	}
	if len(order) != count {
		return nil, fmt.Errorf("feature cache: split length mismatch")
	}
	seen := make([]bool, count)
	for _, i := range order {
		if i < 0 || i >= count || seen[i] {
			return nil, fmt.Errorf("feature cache: invalid split order")
		}
		seen[i] = true
	}
	return order, nil
}
