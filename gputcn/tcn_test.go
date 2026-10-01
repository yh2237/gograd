package gputcn

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/yh2237/gograd"
	"github.com/yh2237/gograd/cuda"
)

type fixtureLayer struct {
	Dilation int           `json:"dilation"`
	Weight   [][][]float64 `json:"weight"`
	Bias     []float64     `json:"bias"`
}

type fixtureGradLayer struct {
	Weight [][][]float64 `json:"weight"`
	Bias   []float64     `json:"bias"`
}

type fixture struct {
	Inputs      int           `json:"inputs"`
	Hidden      int           `json:"hidden"`
	Dilations   []int         `json:"dilations"`
	Batch       int           `json:"batch"`
	Time        int           `json:"time"`
	Values      [][][]float64 `json:"values"`
	Targets     [][]float64   `json:"targets"`
	Mask        [][]bool      `json:"mask"`
	LowCents    float64       `json:"low_cents"`
	HighCents   float64       `json:"high_cents"`
	DeltaWeight float64       `json:"delta_weight"`
	Params      struct {
		InputWeight  [][]float64    `json:"input_weight"`
		InputBias    []float64      `json:"input_bias"`
		Layers       []fixtureLayer `json:"layers"`
		OutputWeight [][]float64    `json:"output_weight"`
		OutputBias   []float64      `json:"output_bias"`
	} `json:"params"`
	Predicted [][]float64 `json:"predicted"`
	Grads     struct {
		InputWeight  [][]float64        `json:"input_weight"`
		InputBias    []float64          `json:"input_bias"`
		Layers       []fixtureGradLayer `json:"layers"`
		OutputWeight [][]float64        `json:"output_weight"`
		OutputBias   []float64          `json:"output_bias"`
	} `json:"grads"`
}

func loadFixture(t *testing.T) *fixture {
	t.Helper()
	raw, err := os.ReadFile("../testdata/tcn_step.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var data fixture
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return &data
}

func toFloat32(values []float64) []float32 {
	result := make([]float32, len(values))
	for i, value := range values {
		result[i] = float32(value)
	}
	return result
}

func toFloat64(values []float32) []float64 {
	result := make([]float64, len(values))
	for i, value := range values {
		result[i] = float64(value)
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

func buildModel(t *testing.T, data *fixture) *Model {
	t.Helper()
	model := &Model{
		Inputs:       data.Inputs,
		Hidden:       data.Hidden,
		InputWeight:  mustUpload(t, toFloat32(flatten2D(data.Params.InputWeight))),
		InputBias:    mustUpload(t, toFloat32(data.Params.InputBias)),
		OutputWeight: mustUpload(t, toFloat32(flatten2D(data.Params.OutputWeight))),
		OutputBias:   mustUpload(t, toFloat32(data.Params.OutputBias)),
	}
	for _, layer := range data.Params.Layers {
		model.Layers = append(model.Layers, Layer{
			Weight:   mustUpload(t, toFloat32(flatten3D(layer.Weight))),
			Bias:     mustUpload(t, toFloat32(layer.Bias)),
			Dilation: layer.Dilation,
		})
	}
	return model
}

func maxRelativeDiff(got []float32, expected []float64) float64 {
	maximum := 0.0
	for i := range expected {
		denom := math.Max(1e-3, math.Abs(expected[i]))
		relative := math.Abs(float64(got[i])-expected[i]) / denom
		if relative > maximum {
			maximum = relative
		}
	}
	return maximum
}

func TestForwardMatchesPyTorch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	data := loadFixture(t)
	model := buildModel(t, data)
	defer model.Close()

	got, err := model.Forward(toFloat32(flatten3D(data.Values)), data.Batch, data.Time)
	if err != nil {
		t.Fatal(err)
	}
	want := flatten2D(data.Predicted)
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	diff := maxRelativeDiff(got, want)
	t.Logf("forward max relative diff %.3e", diff)
	if diff > 1e-4 {
		t.Errorf("forward differs from PyTorch float64 by %.3e", diff)
	}
}

func TestBackwardMatchesPyTorch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	data := loadFixture(t)
	model := buildModel(t, data)
	defer model.Close()

	cache, err := model.ForwardCached(toFloat32(flatten3D(data.Values)), data.Batch, data.Time)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Close()
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	predicted, err := DownloadFloat32(cache.Output(), data.Batch*data.Time)
	if err != nil {
		t.Fatal(err)
	}

	predictedTensor := gograd.NewTensor([]int{data.Batch, data.Time}, toFloat64(predicted))
	targetTensor := gograd.NewTensor([]int{data.Batch, data.Time}, flatten2D(data.Targets))
	loss := gograd.SequenceLoss(predictedTensor, targetTensor, data.Mask, gograd.SequenceLossOptions{
		LowCents: data.LowCents, HighCents: data.HighCents, Bounded: true, TargetScale: 1, DeltaWeight: data.DeltaWeight,
	})
	loss.Backward()
	dyBuffer := mustUpload(t, toFloat32(predictedTensor.Grad))
	defer dyBuffer.Free()

	grads, err := model.Backward(cache, dyBuffer)
	if err != nil {
		t.Fatal(err)
	}
	defer grads.Close()
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}

	check := func(name string, buffer *cuda.Buffer, expected []float64) {
		got, err := DownloadFloat32(buffer, len(expected))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		diff := maxRelativeDiff(got, expected)
		t.Logf("%s: max relative diff %.3e", name, diff)
		if diff > 5e-2 {
			t.Errorf("%s: max relative diff %.3e", name, diff)
		}
	}
	check("input_weight", grads.InputWeight, flatten2D(data.Grads.InputWeight))
	check("input_bias", grads.InputBias, data.Grads.InputBias)
	for i := range grads.Layers {
		check(fmt.Sprintf("layer%d_weight", i), grads.Layers[i].Weight, flatten3D(data.Grads.Layers[i].Weight))
		check(fmt.Sprintf("layer%d_bias", i), grads.Layers[i].Bias, data.Grads.Layers[i].Bias)
	}
	check("output_weight", grads.OutputWeight, flatten2D(data.Grads.OutputWeight))
	check("output_bias", grads.OutputBias, data.Grads.OutputBias)
}
