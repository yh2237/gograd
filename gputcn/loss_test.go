package gputcn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd"
)

func TestLossGradMatchesCPU(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const rows, time = 5, 9
	predicted := make([]float32, rows*time)
	target := make([]float64, rows*time)
	mask := make([][]bool, rows)
	for i := range predicted {
		predicted[i] = float32(rng.NormFloat64() * 100)
	}
	for i := range target {
		target[i] = rng.NormFloat64() * 120
	}
	for r := 0; r < rows; r++ {
		mask[r] = make([]bool, time)
		for c := 0; c < time; c++ {
			mask[r][c] = !(r == 0 && c > time-3)
		}
	}
	options := LossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}
	loss, gradient := LossGrad(predicted, target, mask, options)

	predictedTensor := gograd.NewTensor([]int{rows, time}, toFloat64(predicted))
	targetTensor := gograd.NewTensor([]int{rows, time}, append([]float64(nil), target...))
	reference := gograd.SequenceLoss(predictedTensor, targetTensor, mask, gograd.SequenceLossOptions{
		LowCents: options.LowCents, HighCents: options.HighCents, Bounded: options.Bounded,
		TargetScale: options.TargetScale, DeltaWeight: options.DeltaWeight,
	})
	reference.Backward()

	if math.Abs(loss-reference.Data[0]) > 1e-6*math.Max(1, math.Abs(reference.Data[0])) {
		t.Errorf("loss: got %v want %v", loss, reference.Data[0])
	}
	for i := range gradient {
		if diff := math.Abs(float64(gradient[i]) - predictedTensor.Grad[i]); diff > 1e-4 {
			t.Fatalf("gradient[%d]: got %v want %v", i, gradient[i], predictedTensor.Grad[i])
		}
	}
}
