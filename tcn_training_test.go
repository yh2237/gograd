package gograd

import (
	"math"
	"math/rand"
	"testing"
)

func syntheticTrainingCorpus(seed int64, count, inputs int) ([][][]float64, [][]float64) {
	rng := rand.New(rand.NewSource(seed))
	values := make([][][]float64, count)
	targets := make([][]float64, count)
	for i := 0; i < count; i++ {
		length := 8 + rng.Intn(6)
		row := make([][]float64, length)
		target := make([]float64, length)
		for t := 0; t < length; t++ {
			features := make([]float64, inputs)
			for f := range features {
				features[f] = rng.NormFloat64()
			}
			row[t] = features
		}
		for t := 0; t < length; t++ {
			prev, next := t-1, t+1
			if prev < 0 {
				prev = 0
			}
			if next >= length {
				next = length - 1
			}
			target[t] = 12*row[t][0] - 8*row[t][1%inputs] + 4*row[prev][0] - 3*row[next][2%inputs]
		}
		values[i] = row
		targets[i] = target
	}
	return values, targets
}

// paddedBatch pads variable-length rows into dense row-major arrays.
func paddedBatch(values [][][]float64, targets [][]float64, inputs int) (flatValues, flatTargets []float64, mask [][]bool, length int) {
	for _, row := range values {
		if len(row) > length {
			length = len(row)
		}
	}
	rows := len(values)
	flatValues = make([]float64, rows*length*inputs)
	flatTargets = make([]float64, rows*length)
	mask = make([][]bool, rows)
	for row, valueRow := range values {
		mask[row] = make([]bool, length)
		for t := 0; t < length; t++ {
			if t < len(valueRow) {
				copy(flatValues[(row*length+t)*inputs:], valueRow[t])
				flatTargets[row*length+t] = targets[row][t]
				mask[row][t] = true
			}
		}
	}
	return flatValues, flatTargets, mask, length
}

func sliceRecords(values [][][]float64, targets [][]float64, order []int) ([][][]float64, [][]float64) {
	selectedValues := make([][][]float64, len(order))
	selectedTargets := make([][]float64, len(order))
	for i, index := range order {
		selectedValues[i] = values[index]
		selectedTargets[i] = targets[index]
	}
	return selectedValues, selectedTargets
}

func TestTCNTrainingReducesLoss(t *testing.T) {
	const inputs, hidden, batchSize = 4, 8, 8
	valuesRaw, targetRaw := syntheticTrainingCorpus(5, 24, inputs)
	options := SequenceLossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}

	eval := func(model *FrameIntonationTCN) float64 {
		flatValues, flatTargets, mask, length := paddedBatch(valuesRaw, targetRaw, inputs)
		valueTensor := NewTensor([]int{len(valuesRaw), length, inputs}, flatValues)
		targetTensor := NewTensor([]int{len(valuesRaw), length}, flatTargets)
		return SequenceLoss(model.Forward(valueTensor), targetTensor, mask, options).Data[0]
	}

	model := NewFrameIntonationTCN(inputs, hidden, []int{1, 2, 4})
	model.InitializeParameters(5)
	optimizer := NewAdamW(model.Parameters(), 0.01, 1e-5)
	rng := rand.New(rand.NewSource(5))

	initial := eval(model)
	for epoch := 0; epoch < 40; epoch++ {
		order := rng.Perm(len(valuesRaw))
		for start := 0; start < len(order); start += batchSize {
			end := start + batchSize
			if end > len(order) {
				end = len(order)
			}
			selectedValues, selectedTargets := sliceRecords(valuesRaw, targetRaw, order[start:end])
			flatValues, flatTargets, mask, length := paddedBatch(selectedValues, selectedTargets, inputs)
			valueTensor := NewTensor([]int{end - start, length, inputs}, flatValues)
			targetTensor := NewTensor([]int{end - start, length}, flatTargets)
			loss := SequenceLoss(model.Forward(valueTensor), targetTensor, mask, options)
			model.ZeroGrad()
			loss.Backward()
			ClipGradNorm(model.Parameters(), 1.0)
			optimizer.Step()
		}
	}
	final := eval(model)
	t.Logf("initial loss %.4f -> final loss %.4f", initial, final)
	if math.IsNaN(final) {
		t.Fatal("loss became NaN")
	}
	if !(final < initial*0.75) {
		t.Errorf("training did not reduce loss enough: initial %.4f final %.4f", initial, final)
	}
}
