package gputcn

import (
	"math"
	"sort"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
)

// TrainBatch runs one forward, loss, backward and AdamW step on a prepared
// batch and returns the loss.
func (m *Model) TrainBatch(batch *Batch, optimizer *AdamState, learningRate, weightDecay, maxNorm float64, options LossOptions) (float64, error) {
	input, err := UploadFloat32(batch.Values)
	if err != nil {
		return 0, err
	}
	defer input.Free()
	targetValues := make([]float32, len(batch.Targets))
	for i, value := range batch.Targets {
		targetValues[i] = float32(value)
	}
	targetBuffer, err := UploadFloat32(targetValues)
	if err != nil {
		return 0, err
	}
	defer targetBuffer.Free()

	maskBytes := make([]byte, batch.Batch*batch.Time)
	valid, pairs := 0, 0
	for row := 0; row < batch.Batch; row++ {
		for t := 0; t < batch.Time; t++ {
			if batch.Mask[row][t] {
				maskBytes[row*batch.Time+t] = 1
				valid++
			}
			if t > 0 && batch.Mask[row][t-1] && batch.Mask[row][t] {
				pairs++
			}
		}
	}
	maskBuffer, err := cuda.Alloc(len(maskBytes))
	if err != nil {
		return 0, err
	}
	defer maskBuffer.Free()
	if err := maskBuffer.CopyFromHost(maskBytes); err != nil {
		return 0, err
	}
	gradientBuffer, err := cuda.Alloc(batch.Batch * batch.Time * 4)
	if err != nil {
		return 0, err
	}
	defer gradientBuffer.Free()
	lossBuffer, err := cuda.Alloc(4)
	if err != nil {
		return 0, err
	}
	defer lossBuffer.Free()
	if err := lossBuffer.Memset(0, 4); err != nil {
		return 0, err
	}

	cache, err := m.ForwardCachedDevice(input, batch.Batch, batch.Time)
	if err != nil {
		return 0, err
	}
	if err := kernels.SequenceLossGrad(cache.Output(), targetBuffer, maskBuffer, gradientBuffer, lossBuffer,
		batch.Batch, batch.Time, valid, pairs, options.Bounded, options.LowCents, options.HighCents, options.DeltaWeight); err != nil {
		cache.Close()
		return 0, err
	}
	grads, err := m.Backward(cache, gradientBuffer)
	cache.Close()
	if err != nil {
		return 0, err
	}
	defer grads.Close()
	if err := m.ApplyAdamW(grads, optimizer, learningRate, weightDecay, maxNorm); err != nil {
		return 0, err
	}
	lossValues, err := DownloadFloat32(lossBuffer, 1)
	if err != nil {
		return 0, err
	}
	return float64(lossValues[0]), nil
}

// EvaluateBatch returns the mean loss and the raw MAE in cents after per-row
// centering and clamping, without updating parameters.
func (m *Model) EvaluateBatch(batch *Batch, options LossOptions) (loss, mae float64, err error) {
	input, err := UploadFloat32(batch.Values)
	if err != nil {
		return 0, 0, err
	}
	defer input.Free()
	cache, err := m.ForwardCachedDevice(input, batch.Batch, batch.Time)
	if err != nil {
		return 0, 0, err
	}
	defer cache.Close()
	if err := cuda.Synchronize(); err != nil {
		return 0, 0, err
	}
	predicted, err := DownloadFloat32(cache.Output(), batch.Batch*batch.Time)
	if err != nil {
		return 0, 0, err
	}
	scale := math.Max(1, options.TargetScale)
	absolute, count := 0.0, 0
	for row := 0; row < batch.Batch; row++ {
		values := make([]float64, 0, batch.Time)
		for t := 0; t < batch.Time; t++ {
			if batch.Mask[row][t] {
				values = append(values, float64(predicted[row*batch.Time+t]))
			}
		}
		center := 0.0
		if len(values) > 0 {
			sort.Float64s(values)
			center = values[(len(values)-1)/2]
		}
		for t := 0; t < batch.Time; t++ {
			if !batch.Mask[row][t] {
				continue
			}
			value := float64(predicted[row*batch.Time+t]) - center
			value = math.Min(math.Max(value, options.LowCents/scale), options.HighCents/scale) * scale
			absolute += math.Abs(value - batch.Targets[row*batch.Time+t]*scale)
			count++
		}
	}
	if count > 0 {
		mae = absolute / float64(count)
	}
	loss, _ = LossGrad(predicted, batch.Targets, batch.Mask, options)
	return loss, mae, nil
}
