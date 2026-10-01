package gputcn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

func randomValues(rng *rand.Rand, count int, scale float64) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32((rng.Float64()*2 - 1) * scale)
	}
	return values
}

func syntheticData(rng *rand.Rand, count, inputs, maxTime int) ([][]float32, [][]float64, [][]bool) {
	values := make([][]float32, count)
	targets := make([][]float64, count)
	mask := make([][]bool, count)
	for i := 0; i < count; i++ {
		length := maxTime - rng.Intn(3)
		row := make([]float32, length*inputs)
		target := make([]float64, length)
		rowMask := make([]bool, length)
		features := make([][]float64, length)
		for t := 0; t < length; t++ {
			feature := make([]float64, inputs)
			for f := range feature {
				feature[f] = rng.NormFloat64()
			}
			features[t] = feature
			for f := range feature {
				row[t*inputs+f] = float32(feature[f])
			}
			rowMask[t] = true
		}
		for t := 0; t < length; t++ {
			prev, next := t-1, t+1
			if prev < 0 {
				prev = 0
			}
			if next >= length {
				next = length - 1
			}
			target[t] = 12*features[t][0] - 8*features[t][1%inputs] + 4*features[prev][0] - 3*features[next][2%inputs]
		}
		values[i] = row
		targets[i] = target
		mask[i] = rowMask
	}
	return values, targets, mask
}

func padBatch(values [][]float32, targets [][]float64, mask [][]bool, inputs int) ([]float32, [][]float64, [][]bool, int) {
	rows := len(values)
	length := 0
	for _, target := range targets {
		if len(target) > length {
			length = len(target)
		}
	}
	padded := make([]float32, rows*length*inputs)
	paddedTargets := make([][]float64, rows)
	paddedMask := make([][]bool, rows)
	for r := 0; r < rows; r++ {
		paddedTargets[r] = make([]float64, length)
		paddedMask[r] = make([]bool, length)
		for t := 0; t < length; t++ {
			if t < len(targets[r]) {
				copy(padded[(r*length+t)*inputs:(r*length+t+1)*inputs], values[r][t*inputs:(t+1)*inputs])
				paddedTargets[r][t] = targets[r][t]
				paddedMask[r][t] = mask[r][t]
			}
		}
	}
	return padded, paddedTargets, paddedMask, length
}

func TestTrainingReducesLoss(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	const inputs, hidden, count, maxTime = 4, 6, 8, 10
	rng := rand.New(rand.NewSource(3))
	values, targets, mask := syntheticData(rng, count, inputs, maxTime)
	input, target, maskBatch, length := padBatch(values, targets, mask, inputs)

	model := &Model{
		Inputs:       inputs,
		Hidden:       hidden,
		InputWeight:  mustUpload(t, randomValues(rng, hidden*inputs, 0.5)),
		InputBias:    mustUpload(t, make([]float32, hidden)),
		OutputWeight: mustUpload(t, randomValues(rng, hidden, 0.5)),
		OutputBias:   mustUpload(t, make([]float32, 1)),
	}
	defer model.Close()
	for _, dilation := range []int{1, 2, 4} {
		model.Layers = append(model.Layers, Layer{
			Weight:   mustUpload(t, randomValues(rng, hidden*hidden*3, 0.4)),
			Bias:     mustUpload(t, make([]float32, hidden)),
			Dilation: dilation,
		})
	}

	options := LossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}
	state := NewAdamState()

	evaluate := func() float64 {
		cache, err := model.ForwardCached(input, count, length)
		if err != nil {
			t.Fatal(err)
		}
		defer cache.Close()
		if err := cuda.Synchronize(); err != nil {
			t.Fatal(err)
		}
		predicted, err := DownloadFloat32(cache.Output(), count*length)
		if err != nil {
			t.Fatal(err)
		}
		loss, _ := LossGrad(predicted, flatten2D(target), maskBatch, options)
		return loss
	}

	initial := evaluate()
	for step := 0; step < 60; step++ {
		cache, err := model.ForwardCached(input, count, length)
		if err != nil {
			t.Fatal(err)
		}
		if err := cuda.Synchronize(); err != nil {
			t.Fatal(err)
		}
		predicted, err := DownloadFloat32(cache.Output(), count*length)
		if err != nil {
			t.Fatal(err)
		}
		_, gradient := LossGrad(predicted, flatten2D(target), maskBatch, options)
		dy := mustUpload(t, gradient)
		grads, err := model.Backward(cache, dy)
		cache.Close()
		dy.Free()
		if err != nil {
			t.Fatal(err)
		}
		if err := model.ApplyAdamW(grads, state, 0.01, 1e-5, 1.0); err != nil {
			t.Fatal(err)
		}
		grads.Close()
	}

	final := evaluate()
	t.Logf("initial loss %.4f -> final loss %.4f", initial, final)
	if math.IsNaN(final) {
		t.Fatal("loss became NaN")
	}
	if !(final < initial*0.8) {
		t.Errorf("training did not reduce loss: initial %.4f final %.4f", initial, final)
	}
}
