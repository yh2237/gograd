// Command tcn-train fits the frame intonation TCN on a small synthetic corpus
// and writes the runtime JSON.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/yh2237/gograd"
)

type record struct {
	values [][]float64
	target []float64
}

// syntheticCorpus builds targets that are a fixed linear function of the
// current frame and its immediate neighbours, so the TCN has a learnable
// temporal signal rather than unpredictable per-utterance phase.
func syntheticCorpus(seed int64, count, inputs int) []record {
	rng := rand.New(rand.NewSource(seed))
	records := make([]record, count)
	feature := func(values [][]float64, index, f int) float64 {
		if index < 0 || index >= len(values) || f >= inputs {
			return 0
		}
		return values[index][f]
	}
	for i := range records {
		length := 12 + rng.Intn(10)
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
			target[t] = 12*feature(row, t, 0) - 8*feature(row, t, 1) +
				5*feature(row, t, 2) + 4*feature(row, t-1, 0) -
				3*feature(row, t+1, 3%inputs)
		}
		records[i] = record{values: row, target: target}
	}
	return records
}

func batch(records []record, order []int, inputs int) ([][][]float64, [][]float64, [][]bool) {
	length := 0
	for _, index := range order {
		if len(records[index].target) > length {
			length = len(records[index].target)
		}
	}
	values := make([][][]float64, len(order))
	targets := make([][]float64, len(order))
	mask := make([][]bool, len(order))
	for row, index := range order {
		values[row] = make([][]float64, length)
		targets[row] = make([]float64, length)
		mask[row] = make([]bool, length)
		for t := 0; t < length; t++ {
			values[row][t] = make([]float64, inputs)
			if t < len(records[index].target) {
				copy(values[row][t], records[index].values[t])
				targets[row][t] = records[index].target[t]
				mask[row][t] = true
			}
		}
	}
	return values, targets, mask
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

func flatten2D(values [][]float64) []float64 {
	flat := make([]float64, 0)
	for _, row := range values {
		flat = append(flat, row...)
	}
	return flat
}

func runEpoch(model *gograd.FrameIntonationTCN, records []record, inputs, batchSize int, rng *rand.Rand, options gograd.SequenceLossOptions, update func()) float64 {
	order := rng.Perm(len(records))
	total := 0.0
	batches := 0
	for offset := 0; offset < len(order); offset += batchSize {
		end := offset + batchSize
		if end > len(order) {
			end = len(order)
		}
		values, targets, mask := batch(records, order[offset:end], inputs)
		length := len(mask[0])
		valueTensor := gograd.NewTensor([]int{end - offset, length, inputs}, flatten3D(values))
		targetTensor := gograd.NewTensor([]int{end - offset, length}, flatten2D(targets))
		predicted := model.Forward(valueTensor)
		loss := gograd.SequenceLoss(predicted, targetTensor, mask, options)
		model.ZeroGrad()
		loss.Backward()
		total += loss.Data[0]
		batches++
		update()
	}
	return total / math.Max(1, float64(batches))
}

func main() {
	epochs := flag.Int("epochs", 120, "training epochs")
	hidden := flag.Int("hidden", 12, "TCN hidden size")
	inputs := flag.Int("inputs", 6, "feature count")
	batchSize := flag.Int("batch-size", 16, "batch size")
	learningRate := flag.Float64("learning-rate", 0.01, "AdamW learning rate")
	deltaWeight := flag.Float64("delta-weight", 0.35, "adjacent-frame delta loss weight")
	datasetSize := flag.Int("dataset-size", 48, "synthetic utterances")
	seed := flag.Int64("seed", 1, "random seed")
	out := flag.String("out", "out/synthetic-tcn.json", "output model JSON")
	flag.Parse()

	records := syntheticCorpus(*seed, *datasetSize, *inputs)
	model := gograd.NewFrameIntonationTCN(*inputs, *hidden, []int{1, 2, 4})
	model.InitializeParameters(*seed)
	optimizer := gograd.NewAdamW(model.Parameters(), *learningRate, 1e-5)
	options := gograd.SequenceLossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: *deltaWeight}

	rng := rand.New(rand.NewSource(*seed))
	update := func() {
		gograd.ClipGradNorm(model.Parameters(), 1.0)
		optimizer.Step()
	}
	initial := runEpoch(model, records, *inputs, *batchSize, rand.New(rand.NewSource(*seed)), options, func() {
		model.ZeroGrad()
	})
	fmt.Printf("initial loss: %.6f\n", initial)
	for epoch := 1; epoch <= *epochs; epoch++ {
		mean := runEpoch(model, records, *inputs, *batchSize, rng, options, update)
		if epoch == 1 || epoch%5 == 0 || epoch == *epochs {
			fmt.Printf("epoch %02d/%d: loss=%.6f\n", epoch, *epochs, mean)
		}
	}

	featureNames := make([]string, *inputs)
	for i := range featureNames {
		featureNames[i] = fmt.Sprintf("f%d", i)
	}
	exported := model.ExportFramePitch(featureNames, 10, -250, 250, 1)
	payload, err := json.MarshalIndent(map[string]any{
		"id":               "synthetic-tcn",
		"version":          8,
		"feature_version":  1,
		"mode":             "intonation_frame_tcn_accent_bounded",
		"language":         "ja",
		"duration_weights": map[string]float64{},
		"frame_pitch":      exported,
	}, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "marshal model:", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "create output dir:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, append(payload, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write model:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (%d features)\n", *out, len(featureNames))
}
