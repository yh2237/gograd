package gputcn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
)

func countMasked(mask [][]bool) (int, int) {
	valid, pairs := 0, 0
	for _, row := range mask {
		for t := range row {
			if row[t] {
				valid++
			}
			if t > 0 && row[t-1] && row[t] {
				pairs++
			}
		}
	}
	return valid, pairs
}

func TestSequenceLossGradGPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	rng := rand.New(rand.NewSource(11))
	const rows, time = 6, 13
	predicted := make([]float32, rows*time)
	target := make([]float64, rows*time)
	mask := make([][]bool, rows)
	maskBytes := make([]byte, rows*time)
	for i := range predicted {
		predicted[i] = float32(rng.NormFloat64() * 120)
	}
	for i := range target {
		target[i] = rng.NormFloat64() * 140
	}
	for r := 0; r < rows; r++ {
		mask[r] = make([]bool, time)
		for c := 0; c < time; c++ {
			keep := !(r == 0 && c > time-3)
			mask[r][c] = keep
			if keep {
				maskBytes[r*time+c] = 1
			}
		}
	}
	targetF32 := make([]float32, rows*time)
	for i, value := range target {
		targetF32[i] = float32(value)
	}
	target64 := toFloat64(targetF32)
	options := LossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}
	hostLoss, hostGrad := LossGrad(predicted, target64, mask, options)
	valid, pairs := countMasked(mask)

	predictedBuffer := mustUpload(t, predicted)
	targetBuffer := mustUpload(t, targetF32)
	defer predictedBuffer.Free()
	defer targetBuffer.Free()
	maskBuffer, err := cuda.Alloc(rows * time)
	if err != nil {
		t.Fatal(err)
	}
	defer maskBuffer.Free()
	if err := maskBuffer.CopyFromHost(maskBytes); err != nil {
		t.Fatal(err)
	}
	gradientBuffer, err := cuda.Alloc(rows * time * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer gradientBuffer.Free()
	lossBuffer, err := cuda.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer lossBuffer.Free()
	if err := lossBuffer.Memset(0, 4); err != nil {
		t.Fatal(err)
	}

	if err := kernels.SequenceLossGrad(predictedBuffer, targetBuffer, maskBuffer, gradientBuffer, lossBuffer, rows, time, valid, pairs, true, -250, 250, 0.35); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	gotGradient, err := DownloadFloat32(gradientBuffer, rows*time)
	if err != nil {
		t.Fatal(err)
	}
	gotLoss, err := DownloadFloat32(lossBuffer, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(gotLoss[0])-hostLoss) > 1e-4*math.Max(1, math.Abs(hostLoss)) {
		t.Errorf("loss: gpu %v host %v", gotLoss[0], hostLoss)
	}
	for i := range hostGrad {
		if diff := math.Abs(float64(gotGradient[i]) - float64(hostGrad[i])); diff > 1e-4 {
			t.Errorf("gradient[%d]: gpu %v host %v", i, gotGradient[i], hostGrad[i])
		}
	}
}

func TestSequenceLossGradGPUSimple(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	const rows, time = 1, 4
	predicted := []float32{1, 2, 3, 4}
	target := []float32{0, 0, 0, 0}
	maskBytes := []byte{1, 1, 1, 1}

	predictedBuffer := mustUpload(t, predicted)
	targetBuffer := mustUpload(t, target)
	defer predictedBuffer.Free()
	defer targetBuffer.Free()
	maskBuffer, err := cuda.Alloc(rows * time)
	if err != nil {
		t.Fatal(err)
	}
	defer maskBuffer.Free()
	if err := maskBuffer.CopyFromHost(maskBytes); err != nil {
		t.Fatal(err)
	}
	gradientBuffer, err := cuda.Alloc(rows * time * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer gradientBuffer.Free()
	lossBuffer, err := cuda.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer lossBuffer.Free()
	if err := lossBuffer.Memset(0, 4); err != nil {
		t.Fatal(err)
	}
	if err := kernels.SequenceLossGrad(predictedBuffer, targetBuffer, maskBuffer, gradientBuffer, lossBuffer, rows, time, 4, 3, false, 0, 0, 0); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Synchronize(); err != nil {
		t.Fatal(err)
	}
	gotGradient, err := DownloadFloat32(gradientBuffer, rows*time)
	if err != nil {
		t.Fatal(err)
	}
	gotLoss, err := DownloadFloat32(lossBuffer, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-0.25, -0.25, 0.25, 0.25}
	for i := range want {
		if math.Abs(float64(gotGradient[i]-want[i])) > 1e-5 {
			t.Errorf("gradient[%d]: got %v want %v", i, gotGradient[i], want[i])
		}
	}
	if math.Abs(float64(gotLoss[0])-0.625) > 1e-5 {
		t.Errorf("loss: got %v want 0.625", gotLoss[0])
	}
}

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
