package gputcn

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

type fixtureLayer struct {
	Dilation int           `json:"dilation"`
	Weight   [][][]float64 `json:"weight"`
	Bias     []float64     `json:"bias"`
}

type fixture struct {
	Inputs    int           `json:"inputs"`
	Hidden    int           `json:"hidden"`
	Dilations []int         `json:"dilations"`
	Batch     int           `json:"batch"`
	Time      int           `json:"time"`
	Values    [][][]float64 `json:"values"`
	Params    struct {
		InputWeight  [][]float64    `json:"input_weight"`
		InputBias    []float64      `json:"input_bias"`
		Layers       []fixtureLayer `json:"layers"`
		OutputWeight [][]float64    `json:"output_weight"`
		OutputBias   []float64      `json:"output_bias"`
	} `json:"params"`
	Predicted [][]float64 `json:"predicted"`
}

func toFloat32(values []float64) []float32 {
	result := make([]float32, len(values))
	for i, value := range values {
		result[i] = float32(value)
	}
	return result
}

func flatten2D(values [][]float64) []float64 {
	flat := make([]float64, 0)
	for _, row := range values {
		flat = append(flat, row...)
	}
	return flat
}

func flatten3D(values [][][]float64) []float64 {
	flat := make([]float64, 0)
	for _, matrix := range values {
		for _, row := range matrix {
			flat = append(flat, row...)
		}
	}
	return flat
}

func mustUpload(t *testing.T, values []float32) *cuda.Buffer {
	t.Helper()
	buffer, err := UploadFloat32(values)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	return buffer
}

func TestForwardMatchesPyTorch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	raw, err := os.ReadFile("../testdata/tcn_step.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var data fixture
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	model := &Model{
		Inputs:       data.Inputs,
		Hidden:       data.Hidden,
		InputWeight:  mustUpload(t, toFloat32(flatten2D(data.Params.InputWeight))),
		InputBias:    mustUpload(t, toFloat32(data.Params.InputBias)),
		OutputWeight: mustUpload(t, toFloat32(flatten2D(data.Params.OutputWeight))),
		OutputBias:   mustUpload(t, toFloat32(data.Params.OutputBias)),
	}
	defer model.Close()
	for _, layer := range data.Params.Layers {
		model.Layers = append(model.Layers, Layer{
			Weight:   mustUpload(t, toFloat32(flatten3D(layer.Weight))),
			Bias:     mustUpload(t, toFloat32(layer.Bias)),
			Dilation: layer.Dilation,
		})
	}

	got, err := model.Forward(toFloat32(flatten3D(data.Values)), data.Batch, data.Time)
	if err != nil {
		t.Fatal(err)
	}
	want := flatten2D(data.Predicted)
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	maxDiff := 0.0
	for i := range want {
		diff := math.Abs(float64(got[i]) - want[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("max abs diff %.3e", maxDiff)
	if maxDiff > 2e-3 {
		t.Errorf("forward differs from PyTorch float64 by %.3e", maxDiff)
	}
}
