package autograd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestSafeTensorStream(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream.safetensors")
	if err := writeSafetensors(path, map[string]safeTensorEntry{
		"block.weight": {Values: []float32{1.25, -2.5, 3.75, 4.5}, Shape: []int{2, 2}},
		"block.bias":   {Values: []float32{-.125, .375}, Shape: []int{2}},
	}, map[string]string{"format": "test"}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSafeTensorFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.Metadata["format"] != "test" || !slices.Equal(s.Names(), []string{"block.bias", "block.weight"}) {
		t.Fatal("invalid index")
	}
	if err := s.PreloadF32("block.bias"); err != nil {
		t.Fatal(err)
	}
	first, _, err := s.ReadF32("block.bias")
	if err != nil {
		t.Fatal(err)
	}
	second, _, err := s.ReadF32("block.bias")
	if err != nil {
		t.Fatal(err)
	}
	if s.CachedBytes() != 8 || &first[0] != &second[0] {
		t.Fatal("resident tensor was reloaded")
	}
	s.Close()
	w, err := Zeros([]int{2, 2}, tensor.CPU, true)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	b, err := Zeros([]int{2}, tensor.CPU, true)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	m := &Module{Children: []NamedModule{{Name: "block", Module: &Module{Parameters: []Parameter{{Name: "weight", Value: w}, {Name: "bias", Value: b}}}}}}
	if err := m.LoadSafeTensorsStream(path); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(w.Data, []float32{1.25, -2.5, 3.75, 4.5}) || !slices.Equal(b.Data, []float32{-.125, .375}) {
		t.Fatal("streamed values differ")
	}
}

// IRODORI_CHECKPOINT points to the read-only HF model.safetensors. This test
// is optional for ordinary CI; its fixture was dumped from that checkpoint by
// tools/gen_irodori_checkpoint_fixture.py using safetensors/PyTorch.
func TestIrodoriCheckpointTensorParity(t *testing.T) {
	path := os.Getenv("IRODORI_CHECKPOINT")
	if path == "" {
		t.Skip("set IRODORI_CHECKPOINT for real checkpoint parity")
	}
	data, err := os.ReadFile("../testdata/irodori_checkpoint_tensor.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Name        string    `json:"name"`
		Shape       []int     `json:"shape"`
		Values      []float32 `json:"values"`
		TensorCount int       `json:"tensor_count"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSafeTensorFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if len(s.Names()) != fixture.TensorCount {
		t.Fatalf("got %d tensors, want %d", len(s.Names()), fixture.TensorCount)
	}
	values, shape, err := s.ReadF32(fixture.Name)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(shape, fixture.Shape) || len(values) != len(fixture.Values) {
		t.Fatal("shape mismatch")
	}
	var maxAbs, maxRel float64
	for i, v := range values {
		diff := math.Abs(float64(v - fixture.Values[i]))
		maxAbs = max(maxAbs, diff)
		maxRel = max(maxRel, diff/math.Max(math.Abs(float64(fixture.Values[i])), 1e-6))
	}
	t.Logf("checkpoint tensor %s: max_abs=%.9g max_rel=%.9g", fixture.Name, maxAbs, maxRel)
	if maxAbs != 0 {
		t.Fatalf("checkpoint mismatch")
	}
}
