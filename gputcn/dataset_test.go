package gputcn

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

func syntheticDataset(features, count int, seed int64) *Dataset {
	rng := rand.New(rand.NewSource(seed))
	names := make([]string, features)
	for i := range names {
		names[i] = string(rune('a' + i))
	}
	dataset := &Dataset{Version: 1, FrameMS: 10, FeatureNames: names}
	for i := 0; i < count; i++ {
		length := 8 + rng.Intn(6)
		record := Record{ID: string(rune('A'+i%26)) + string(rune('a'+i/26)), Length: length,
			Targets: make([]float64, length), Mask: make([]bool, length)}
		values := make([][]float64, length)
		for t := 0; t < length; t++ {
			values[t] = make([]float64, features)
			for f := range values[t] {
				values[t][f] = rng.NormFloat64()
				record.Rows = append(record.Rows, int32(t))
				record.Cols = append(record.Cols, int32(f))
				record.Vals = append(record.Vals, float32(values[t][f]))
			}
			record.Mask[t] = true
		}
		for t := 0; t < length; t++ {
			prev, next := t-1, t+1
			if prev < 0 {
				prev = 0
			}
			if next >= length {
				next = length - 1
			}
			record.Targets[t] = 12*values[t][0] - 8*values[t][1%features] + 4*values[prev][0] - 3*values[next][2%features]
		}
		dataset.Records = append(dataset.Records, record)
	}
	return dataset
}

func meanMAE(t *testing.T, model *Model, dataset *Dataset, indices []int, options LossOptions) float64 {
	t.Helper()
	batch := dataset.BuildBatch(indices)
	_, mae, err := model.EvaluateBatch(batch, options)
	if err != nil {
		t.Fatal(err)
	}
	return mae
}

func TestDatasetTrainingReducesMAE(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	dataset := syntheticDataset(4, 24, 9)
	trainIndices, validationIndices := dataset.Split()
	if len(trainIndices) == 0 || len(validationIndices) == 0 {
		t.Fatal("split produced an empty side")
	}
	model, err := NewFrameIntonationTCN(dataset.Features(), 12, []int{1, 2, 4}, 9)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	optimizer := NewAdamState()
	defer optimizer.Close()
	options := LossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}

	initial := meanMAE(t, model, dataset, validationIndices, options)
	rng := rand.New(rand.NewSource(9))
	for epoch := 0; epoch < 40; epoch++ {
		order := rng.Perm(len(trainIndices))
		for start := 0; start < len(order); start += 8 {
			end := min(start+8, len(order))
			indices := make([]int, end-start)
			for i, position := range order[start:end] {
				indices[i] = trainIndices[position]
			}
			if _, err := model.TrainBatch(dataset.BuildBatch(indices), optimizer, 0.01, 1e-5, 1.0, options); err != nil {
				t.Fatal(err)
			}
		}
	}
	final := meanMAE(t, model, dataset, validationIndices, options)
	t.Logf("dataset MAE %.3f -> %.3f cents", initial, final)
	if math.IsNaN(final) {
		t.Fatal("MAE became NaN")
	}
	if !(final < initial) {
		t.Errorf("training did not reduce MAE: initial %.3f final %.3f", initial, final)
	}
}
