package gograd

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type fixtureLayer struct {
	Dilation int           `json:"dilation"`
	Weight   [][][]float64 `json:"weight"`
	Bias     []float64     `json:"bias"`
}

type fixtureParams struct {
	InputWeight  [][]float64    `json:"input_weight"`
	InputBias    []float64      `json:"input_bias"`
	Layers       []fixtureLayer `json:"layers"`
	OutputWeight [][]float64    `json:"output_weight"`
	OutputBias   []float64      `json:"output_bias"`
}

type fixtureGradLayer struct {
	Weight [][][]float64 `json:"weight"`
	Bias   []float64     `json:"bias"`
}

type fixtureGrads struct {
	InputWeight  [][]float64        `json:"input_weight"`
	InputBias    []float64          `json:"input_bias"`
	Layers       []fixtureGradLayer `json:"layers"`
	OutputWeight [][]float64        `json:"output_weight"`
	OutputBias   []float64          `json:"output_bias"`
}

type fixture struct {
	Inputs        int           `json:"inputs"`
	Hidden        int           `json:"hidden"`
	Dilations     []int         `json:"dilations"`
	Batch         int           `json:"batch"`
	Time          int           `json:"time"`
	LowCents      float64       `json:"low_cents"`
	HighCents     float64       `json:"high_cents"`
	DeltaWeight   float64       `json:"delta_weight"`
	Values        [][][]float64 `json:"values"`
	Targets       [][]float64   `json:"targets"`
	Mask          [][]bool      `json:"mask"`
	Params        fixtureParams `json:"params"`
	Predicted     [][]float64   `json:"predicted"`
	Loss          float64       `json:"loss"`
	Grads         fixtureGrads  `json:"grads"`
	LearningRate  float64       `json:"learning_rate"`
	WeightDecay   float64       `json:"weight_decay"`
	MaxNorm       float64       `json:"max_norm"`
	TotalNorm     float64       `json:"total_norm"`
	ClippedGrads  fixtureGrads  `json:"clipped_grads"`
	UpdatedParams fixtureParams `json:"updated_params"`
}

func loadFixture(t *testing.T, path string) *fixture {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var data fixture
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return &data
}

func setParams(t *testing.T, model *FrameIntonationTCN, params fixtureParams) {
	t.Helper()
	copy(model.InputWeight.Data, flatten2D(params.InputWeight))
	copy(model.InputBias.Data, params.InputBias)
	for i := range model.Layers {
		source := params.Layers[i]
		if source.Dilation != model.Layers[i].Dilation {
			t.Fatalf("layer %d dilation mismatch", i)
		}
		copy(model.Layers[i].Weight.Data, flatten3D(source.Weight))
		copy(model.Layers[i].Bias.Data, source.Bias)
	}
	copy(model.OutputWeight.Data, flatten2D(params.OutputWeight))
	copy(model.OutputBias.Data, params.OutputBias)
}

func checkParams(t *testing.T, name string, model *FrameIntonationTCN, params fixtureParams, tolerance float64) {
	t.Helper()
	checkClose(t, name+" input_weight", model.InputWeight.Data, flatten2D(params.InputWeight), tolerance)
	checkClose(t, name+" input_bias", model.InputBias.Data, params.InputBias, tolerance)
	for i := range model.Layers {
		checkClose(t, name+" layer weight", model.Layers[i].Weight.Data, flatten3D(params.Layers[i].Weight), tolerance)
		checkClose(t, name+" layer bias", model.Layers[i].Bias.Data, params.Layers[i].Bias, tolerance)
	}
	checkClose(t, name+" output_weight", model.OutputWeight.Data, flatten2D(params.OutputWeight), tolerance)
	checkClose(t, name+" output_bias", model.OutputBias.Data, params.OutputBias, tolerance)
}

func checkGrads(t *testing.T, name string, model *FrameIntonationTCN, grads fixtureGrads, tolerance float64) {
	t.Helper()
	checkClose(t, name+" input_weight", model.InputWeight.Grad, flatten2D(grads.InputWeight), tolerance)
	checkClose(t, name+" input_bias", model.InputBias.Grad, grads.InputBias, tolerance)
	for i := range model.Layers {
		checkClose(t, name+" layer weight", model.Layers[i].Weight.Grad, flatten3D(grads.Layers[i].Weight), tolerance)
		checkClose(t, name+" layer bias", model.Layers[i].Bias.Grad, grads.Layers[i].Bias, tolerance)
	}
	checkClose(t, name+" output_weight", model.OutputWeight.Grad, flatten2D(grads.OutputWeight), tolerance)
	checkClose(t, name+" output_bias", model.OutputBias.Grad, grads.OutputBias, tolerance)
}

func checkClose(t *testing.T, name string, actual, expected []float64, tolerance float64) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("%s: length %d != %d", name, len(actual), len(expected))
	}
	maxDiff := 0.0
	for i := range actual {
		diff := math.Abs(actual[i] - expected[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	limit := tolerance * math.Max(1, maxAbs(expected))
	t.Logf("%s: max abs diff %.3e (limit %.3e)", name, maxDiff, limit)
	if maxDiff > limit {
		t.Errorf("%s: max abs diff %.3e exceeds %.3e", name, maxDiff, limit)
	}
}

func maxAbs(values []float64) float64 {
	maximum := 0.0
	for _, value := range values {
		if abs := math.Abs(value); abs > maximum {
			maximum = abs
		}
	}
	return maximum
}

func TestTCNSingleStepMatchesPyTorch(t *testing.T) {
	data := loadFixture(t, "testdata/tcn_step.json")
	model := NewFrameIntonationTCN(data.Inputs, data.Hidden, data.Dilations)
	setParams(t, model, data.Params)

	values := NewTensor([]int{data.Batch, data.Time, data.Inputs}, flatten3D(data.Values))
	targets := NewTensor([]int{data.Batch, data.Time}, flatten2D(data.Targets))

	predicted := model.Forward(values)
	checkClose(t, "forward", predicted.Data, flatten2D(data.Predicted), 1e-9)

	loss := SequenceLoss(predicted, targets, data.Mask, SequenceLossOptions{
		LowCents:    data.LowCents,
		HighCents:   data.HighCents,
		Bounded:     true,
		TargetScale: 1,
		DeltaWeight: data.DeltaWeight,
	})
	if diff := math.Abs(loss.Data[0] - data.Loss); diff > 1e-9*math.Max(1, math.Abs(data.Loss)) {
		t.Errorf("loss: got %.10f want %.10f", loss.Data[0], data.Loss)
	}

	loss.Backward()
	checkGrads(t, "grad", model, data.Grads, 1e-7)

	totalNorm := ClipGradNorm(model.Parameters(), data.MaxNorm)
	if diff := math.Abs(totalNorm - data.TotalNorm); diff > 1e-7*math.Max(1, data.TotalNorm) {
		t.Errorf("total norm: got %.10f want %.10f", totalNorm, data.TotalNorm)
	}
	checkGrads(t, "clipped grad", model, data.ClippedGrads, 1e-7)

	optimizer := NewAdamW(model.Parameters(), data.LearningRate, data.WeightDecay)
	optimizer.Step()
	checkParams(t, "updated", model, data.UpdatedParams, 1e-9)
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
