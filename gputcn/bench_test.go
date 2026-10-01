package gputcn

import (
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

const (
	benchInputs = 40
	benchHidden = 24
	benchBatch  = 64
	benchTime   = 200
)

func benchmarkSetup(b *testing.B) (*Model, []float32, [][]float64, [][]bool, *cuda.Buffer) {
	b.Helper()
	if !cuda.Available() {
		b.Skip("cuda unavailable")
	}
	rng := rand.New(rand.NewSource(1))
	values, targets, mask := syntheticData(rng, benchBatch, benchInputs, benchTime)
	input, target, maskBatch, length := padBatch(values, targets, mask, benchInputs)
	model := &Model{
		Inputs:       benchInputs,
		Hidden:       benchHidden,
		InputWeight:  uploadBench(b, randomValues(rng, benchHidden*benchInputs, 0.5)),
		InputBias:    uploadBench(b, make([]float32, benchHidden)),
		OutputWeight: uploadBench(b, randomValues(rng, benchHidden, 0.5)),
		OutputBias:   uploadBench(b, make([]float32, 1)),
	}
	for _, dilation := range []int{1, 2, 4, 8} {
		model.Layers = append(model.Layers, Layer{
			Weight:   uploadBench(b, randomValues(rng, benchHidden*benchHidden*3, 0.4)),
			Bias:     uploadBench(b, make([]float32, benchHidden)),
			Dilation: dilation,
		})
	}
	dy, err := UploadFloat32(make([]float32, benchBatch*length))
	if err != nil {
		b.Fatal(err)
	}
	return model, input, target, maskBatch, dy
}

func BenchmarkForward(b *testing.B) {
	model, input, _, _, dy := benchmarkSetup(b)
	defer model.Close()
	defer dy.Free()
	length := len(input) / (benchBatch * benchInputs)
	warmup, err := model.ForwardCached(input, benchBatch, length)
	if err != nil {
		b.Fatal(err)
	}
	warmup.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache, err := model.ForwardCached(input, benchBatch, length)
		if err != nil {
			b.Fatal(err)
		}
		cache.Close()
	}
}

func BenchmarkBackward(b *testing.B) {
	model, input, _, _, dy := benchmarkSetup(b)
	defer model.Close()
	defer dy.Free()
	length := len(input) / (benchBatch * benchInputs)
	cache, err := model.ForwardCached(input, benchBatch, length)
	if err != nil {
		b.Fatal(err)
	}
	defer cache.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grads, err := model.Backward(cache, dy)
		if err != nil {
			b.Fatal(err)
		}
		grads.Close()
	}
}

func BenchmarkTrainStep(b *testing.B) {
	model, input, target, mask, _ := benchmarkSetup(b)
	defer model.Close()
	length := len(input) / (benchBatch * benchInputs)
	options := LossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}
	state := NewAdamState()
	warmup, err := model.ForwardCached(input, benchBatch, length)
	if err != nil {
		b.Fatal(err)
	}
	warmupDy, err := UploadFloat32(make([]float32, benchBatch*length))
	if err != nil {
		b.Fatal(err)
	}
	warmupGrads, err := model.Backward(warmup, warmupDy)
	if err != nil {
		b.Fatal(err)
	}
	warmupGrads.Close()
	warmupDy.Free()
	warmup.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache, err := model.ForwardCached(input, benchBatch, length)
		if err != nil {
			b.Fatal(err)
		}
		predicted, err := DownloadFloat32(cache.Output(), benchBatch*length)
		if err != nil {
			b.Fatal(err)
		}
		_, gradient := LossGrad(predicted, flatten2D(target), mask, options)
		dy, err := UploadFloat32(gradient)
		if err != nil {
			b.Fatal(err)
		}
		grads, err := model.Backward(cache, dy)
		cache.Close()
		dy.Free()
		if err != nil {
			b.Fatal(err)
		}
		if err := model.ApplyAdamW(grads, state, 0.005, 1e-5, 1.0); err != nil {
			b.Fatal(err)
		}
		grads.Close()
	}
}

func uploadBench(b *testing.B, values []float32) *cuda.Buffer {
	b.Helper()
	buffer, err := UploadFloat32(values)
	if err != nil {
		b.Fatal(err)
	}
	return buffer
}
