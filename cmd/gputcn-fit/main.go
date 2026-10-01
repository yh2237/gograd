// Command gputcn-fit trains the frame intonation TCN on a prepared dataset
// (frame features, targets and mask) and writes the runtime JSON. The dataset
// is produced elsewhere; this tool only reads it and runs the GPU training loop.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/gputcn"
)

func syntheticDataset(features, count, seed int) *gputcn.Dataset {
	rng := rand.New(rand.NewSource(int64(seed)))
	names := make([]string, features)
	for i := range names {
		names[i] = fmt.Sprintf("f%d", i)
	}
	dataset := &gputcn.Dataset{Version: 1, FrameMS: 10, FeatureNames: names}
	for i := 0; i < count; i++ {
		length := 12 + rng.Intn(10)
		record := gputcn.Record{ID: fmt.Sprintf("utt-%03d", i), Length: length,
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
			prev, next := max(0, t-1), min(length-1, t+1)
			record.Targets[t] = 12*values[t][0] - 8*values[t][1] + 4*values[prev][0] - 3*values[next][2%features]
		}
		dataset.Records = append(dataset.Records, record)
	}
	return dataset
}

func writeDataset(path string, dataset *gputcn.Dataset) error {
	payload, err := json.Marshal(dataset)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, payload, 0o644)
}

func main() {
	datasetPath := flag.String("dataset", "", "prepared dataset JSON")
	writeSynthetic := flag.String("write-synthetic", "", "write a synthetic dataset to this path and exit")
	syntheticCount := flag.Int("synthetic-records", 64, "records in the synthetic dataset")
	epochs := flag.Int("epochs", 30, "training epochs")
	hidden := flag.Int("hidden", 24, "TCN hidden size")
	batchSize := flag.Int("batch-size", 16, "records per batch")
	learningRate := flag.Float64("learning-rate", 0.002, "AdamW learning rate")
	deltaWeight := flag.Float64("delta-weight", 0.35, "adjacent-frame delta loss weight")
	lowCents := flag.Float64("low-cents", -250, "target lower bound")
	highCents := flag.Float64("high-cents", 250, "target upper bound")
	seed := flag.Int64("seed", 1, "random seed")
	out := flag.String("out", "out/frame-tcn.json", "output model JSON")
	flag.Parse()

	if !cuda.Available() {
		fmt.Fprintln(os.Stderr, "cuda unavailable")
		os.Exit(1)
	}

	if *writeSynthetic != "" {
		dataset := syntheticDataset(6, *syntheticCount, int(*seed))
		if err := writeDataset(*writeSynthetic, dataset); err != nil {
			fmt.Fprintln(os.Stderr, "write synthetic:", err)
			os.Exit(1)
		}
		fmt.Printf("wrote synthetic dataset %s (%d records)\n", *writeSynthetic, len(dataset.Records))
		return
	}
	if *datasetPath == "" {
		fmt.Fprintln(os.Stderr, "either -dataset or -write-synthetic is required")
		os.Exit(1)
	}

	dataset, err := gputcn.LoadDataset(*datasetPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load dataset:", err)
		os.Exit(1)
	}
	trainIndices, validationIndices := dataset.Split()
	if len(trainIndices) == 0 || len(validationIndices) == 0 {
		fmt.Fprintln(os.Stderr, "dataset needs at least one train and one validation record")
		os.Exit(1)
	}

	model, err := gputcn.NewFrameIntonationTCN(dataset.Features(), *hidden, []int{1, 2, 4, 8}, *seed)
	if err != nil {
		fmt.Fprintln(os.Stderr, "model:", err)
		os.Exit(1)
	}
	defer model.Close()
	optimizer := gputcn.NewAdamState()
	defer optimizer.Close()
	options := gputcn.LossOptions{LowCents: *lowCents, HighCents: *highCents, Bounded: true, TargetScale: 1, DeltaWeight: *deltaWeight}

	validationBatches := make([][]int, 0)
	for start := 0; start < len(validationIndices); start += *batchSize {
		end := min(start+*batchSize, len(validationIndices))
		validationBatches = append(validationBatches, validationIndices[start:end])
	}

	rng := rand.New(rand.NewSource(*seed))
	bestMAE := math.Inf(1)
	bestEpoch := 0
	for epoch := 1; epoch <= *epochs; epoch++ {
		order := rng.Perm(len(trainIndices))
		total, batches := 0.0, 0
		for start := 0; start < len(order); start += *batchSize {
			end := min(start+*batchSize, len(order))
			indices := make([]int, end-start)
			for i, position := range order[start:end] {
				indices[i] = trainIndices[position]
			}
			batch := dataset.BuildBatch(indices)
			loss, err := model.TrainBatch(batch, optimizer, *learningRate, 1e-5, 1.0, options)
			if err != nil {
				fmt.Fprintln(os.Stderr, "train:", err)
				os.Exit(1)
			}
			total += loss
			batches++
		}
		maeSum, maeBatches := 0.0, 0
		for _, indices := range validationBatches {
			_, mae, err := model.EvaluateBatch(dataset.BuildBatch(indices), options)
			if err != nil {
				fmt.Fprintln(os.Stderr, "evaluate:", err)
				os.Exit(1)
			}
			maeSum += mae
			maeBatches++
		}
		mae := maeSum / math.Max(1, float64(maeBatches))
		if mae < bestMAE {
			bestMAE, bestEpoch = mae, epoch
		}
		if epoch == 1 || epoch%5 == 0 || epoch == *epochs {
			fmt.Printf("epoch %02d/%d: loss=%.4f validation_mae=%.3f cents (best %.3f at %d)\n",
				epoch, *epochs, total/math.Max(1, float64(batches)), mae, bestMAE, bestEpoch)
		}
	}

	exported, err := model.ExportFramePitch(dataset.FeatureNames, dataset.FrameMS, *lowCents, *highCents, 1)
	if err != nil {
		fmt.Fprintln(os.Stderr, "export:", err)
		os.Exit(1)
	}
	payload, err := json.MarshalIndent(map[string]any{
		"id":               "frame-tcn-fit",
		"version":          8,
		"feature_version":  1,
		"mode":             "intonation_frame_tcn_accent_bounded",
		"language":         "ja",
		"duration_weights": map[string]float64{},
		"frame_pitch":      exported,
		"metrics":          map[string]float64{"validation_mae_cents": bestMAE},
		"training":         map[string]any{"epochs": *epochs, "hidden": *hidden, "batch_size": *batchSize, "learning_rate": *learningRate, "seed": *seed, "best_epoch": bestEpoch, "records": len(dataset.Records)},
	}, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "marshal:", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, append(payload, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (features %d, train %d, validation %d, best validation MAE %.3f cents)\n",
		*out, dataset.Features(), len(trainIndices), len(validationIndices), bestMAE)
}
