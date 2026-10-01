package gograd

import (
	"math"
	"math/rand"
	"testing"
)

// numericGrad estimates d loss / d param[i] with central differences. The
// builder must construct a fresh graph each call.
func numericGrad(loss func() *Tensor, param *Tensor, eps float64) []float64 {
	grad := make([]float64, param.Numel())
	original := append([]float64(nil), param.Data...)
	for i := range param.Data {
		param.Data[i] = original[i] + eps
		plus := loss().Data[0]
		param.Data[i] = original[i] - eps
		minus := loss().Data[0]
		param.Data[i] = original[i]
		grad[i] = (plus - minus) / (2 * eps)
	}
	return grad
}

func compareGrad(t *testing.T, name string, analytic, numeric []float64, tolerance float64) {
	t.Helper()
	maxRel := 0.0
	for i := range analytic {
		denom := math.Max(1e-3, math.Abs(numeric[i]))
		rel := math.Abs(analytic[i]-numeric[i]) / denom
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("%s: max relative diff %.3e", name, maxRel)
	if maxRel > tolerance {
		t.Errorf("%s: max relative diff %.3e exceeds %.3e", name, maxRel, tolerance)
	}
}

func makeSyntheticRecords(seed int64, count, inputs int) (values [][][]float64, targets [][]float64) {
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < count; i++ {
		length := 12 + rng.Intn(10)
		row := make([][]float64, length)
		target := make([]float64, length)
		for t := 0; t < length; t++ {
			features := make([]float64, inputs)
			for f := range features {
				features[f] = rng.NormFloat64()
			}
			row[t] = features
			target[t] = 120*math.Sin(0.35*float64(t)+0.7*float64(i)) + 15*features[0] - 10*features[1%inputs]
		}
		values = append(values, row)
		targets = append(targets, target)
	}
	return values, targets
}

func TestSequenceLossGradientMatchesNumeric(t *testing.T) {
	const inputs, hidden = 4, 5
	valuesRaw, targetRaw := makeSyntheticRecords(11, 3, inputs)
	model := NewFrameIntonationTCN(inputs, hidden, []int{1, 2, 4})
	model.InitializeParameters(3)

	batch := len(valuesRaw)
	time := 0
	for _, row := range valuesRaw {
		if len(row) > time {
			time = len(row)
		}
	}
	valuesData := make([]float64, batch*time*inputs)
	targetData := make([]float64, batch*time)
	mask := make([][]bool, batch)
	for b, row := range valuesRaw {
		mask[b] = make([]bool, time)
		for t := 0; t < time; t++ {
			if t < len(row) {
				mask[b][t] = true
				copy(valuesData[(b*time+t)*inputs:], row[t])
				targetData[b*time+t] = targetRaw[b][t]
			}
		}
	}

	loss := func() *Tensor {
		values := NewTensor([]int{batch, time, inputs}, append([]float64(nil), valuesData...))
		targets := NewTensor([]int{batch, time}, append([]float64(nil), targetData...))
		predicted := model.Forward(values)
		return SequenceLoss(predicted, targets, mask, SequenceLossOptions{
			LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35,
		})
	}

	loss().Backward()
	const eps = 1e-6
	for _, param := range model.Parameters() {
		analytic := append([]float64(nil), param.Grad...)
		numeric := numericGrad(loss, param, eps)
		compareGrad(t, "sequence loss", analytic, numeric, 1e-4)
	}
}
