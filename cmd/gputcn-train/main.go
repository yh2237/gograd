// Command gputcn-train fits the frame intonation TCN on a small synthetic
// corpus with the GPU forward and backward passes, prints per-phase timings,
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
	"time"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/gputcn"
	"github.com/yh2237/gograd/kernels"
)

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

func padBatch(values [][]float32, targets [][]float64, mask [][]bool, inputs int) ([]float32, []float64, [][]bool, int) {
	rows := len(values)
	length := 0
	for _, target := range targets {
		if len(target) > length {
			length = len(target)
		}
	}
	padded := make([]float32, rows*length*inputs)
	paddedTargets := make([]float64, rows*length)
	paddedMask := make([][]bool, rows)
	for r := 0; r < rows; r++ {
		paddedMask[r] = make([]bool, length)
		for t := 0; t < length; t++ {
			if t < len(targets[r]) {
				copy(padded[(r*length+t)*inputs:(r*length+t+1)*inputs], values[r][t*inputs:(t+1)*inputs])
				paddedTargets[r*length+t] = targets[r][t]
				paddedMask[r][t] = mask[r][t]
			}
		}
	}
	return padded, paddedTargets, paddedMask, length
}

func randomValues(rng *rand.Rand, count int, scale float64) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32((rng.Float64()*2 - 1) * scale)
	}
	return values
}

func upload(values []float32) (*cuda.Buffer, error) { return gputcn.UploadFloat32(values) }

func main() {
	steps := flag.Int("steps", 200, "training steps")
	hidden := flag.Int("hidden", 24, "TCN hidden size (must match the prepared features)")
	inputs := flag.Int("inputs", 40, "feature count")
	batch := flag.Int("batch", 64, "utterances per batch")
	maxTime := flag.Int("time", 200, "maximum frames per utterance")
	learningRate := flag.Float64("learning-rate", 0.005, "AdamW learning rate")
	seed := flag.Int64("seed", 1, "random seed")
	out := flag.String("out", "out/gputcn-synthetic.json", "output model JSON")
	graphMode := flag.Bool("graph", false, "capture the step in a CUDA graph and replay it")
	flag.Parse()

	if !cuda.Available() {
		fmt.Fprintln(os.Stderr, "cuda unavailable")
		os.Exit(1)
	}
	rng := rand.New(rand.NewSource(*seed))
	values, targets, mask := syntheticData(rng, *batch, *inputs, *maxTime)
	input, target, maskBatch, length := padBatch(values, targets, mask, *inputs)

	model := &gputcn.Model{
		Inputs:       *inputs,
		Hidden:       *hidden,
		InputWeight:  mustUpload(rng, *hidden*(*inputs), 0.5),
		InputBias:    mustUpload(rng, *hidden, 0),
		OutputWeight: mustUpload(rng, *hidden, 0.5),
		OutputBias:   mustUpload(rng, 1, 0),
	}
	defer model.Close()
	for _, dilation := range []int{1, 2, 4, 8} {
		model.Layers = append(model.Layers, gputcn.Layer{
			Weight:   mustUpload(rng, *hidden*(*hidden)*3, 0.4),
			Bias:     mustUpload(rng, *hidden, 0),
			Dilation: dilation,
		})
	}

	options := gputcn.LossOptions{LowCents: -250, HighCents: 250, Bounded: true, TargetScale: 1, DeltaWeight: 0.35}
	state := gputcn.NewAdamState()
	rows := *batch * length

	targetF32 := make([]float32, rows)
	for i, value := range target {
		targetF32[i] = float32(value)
	}
	targetBuffer, err := upload(targetF32)
	if err != nil {
		fmt.Fprintln(os.Stderr, "upload target:", err)
		os.Exit(1)
	}
	defer targetBuffer.Free()
	maskBytes := make([]byte, rows)
	totalValid, totalPairs := 0, 0
	for r := 0; r < *batch; r++ {
		for c := 0; c < length; c++ {
			if maskBatch[r][c] {
				maskBytes[r*length+c] = 1
				totalValid++
			}
			if c > 0 && maskBatch[r][c-1] && maskBatch[r][c] {
				totalPairs++
			}
		}
	}
	maskBuffer, err := cuda.Alloc(rows)
	if err != nil {
		fmt.Fprintln(os.Stderr, "alloc mask:", err)
		os.Exit(1)
	}
	defer maskBuffer.Free()
	if err := maskBuffer.CopyFromHost(maskBytes); err != nil {
		fmt.Fprintln(os.Stderr, "upload mask:", err)
		os.Exit(1)
	}
	gradientBuffer, err := cuda.Alloc(rows * 4)
	if err != nil {
		fmt.Fprintln(os.Stderr, "alloc gradient:", err)
		os.Exit(1)
	}
	defer gradientBuffer.Free()
	lossBuffer, err := cuda.Alloc(4)
	if err != nil {
		fmt.Fprintln(os.Stderr, "alloc loss:", err)
		os.Exit(1)
	}
	defer lossBuffer.Free()

	stepsCount := float64(*steps)
	initial := 0.0

	if *graphMode {
		inputBuffer, err := upload(input)
		if err != nil {
			fmt.Fprintln(os.Stderr, "upload input:", err)
			os.Exit(1)
		}
		defer inputBuffer.Free()
		stepOnce := func() error {
			cache, err := model.ForwardCachedDevice(inputBuffer, *batch, length)
			if err != nil {
				return err
			}
			if err := lossBuffer.MemsetAsync(0, 4); err != nil {
				cache.Close()
				return err
			}
			if err := kernels.SequenceLossGrad(cache.Output(), targetBuffer, maskBuffer, gradientBuffer, lossBuffer,
				*batch, length, totalValid, totalPairs, options.Bounded, options.LowCents, options.HighCents, options.DeltaWeight); err != nil {
				cache.Close()
				return err
			}
			grads, err := model.Backward(cache, gradientBuffer)
			if err != nil {
				cache.Close()
				return err
			}
			err = model.ApplyAdamW(grads, state, *learningRate, 1e-5, 1.0)
			cache.Close()
			grads.Close()
			return err
		}

		stream, err := cuda.NewStream()
		if err != nil {
			fmt.Fprintln(os.Stderr, "stream:", err)
			os.Exit(1)
		}
		defer stream.Destroy()
		cuda.SetCurrentStream(stream)
		defer cuda.SetCurrentStream(nil)
		if err := model.SetStream(stream); err != nil {
			fmt.Fprintln(os.Stderr, "bind stream:", err)
			os.Exit(1)
		}
		for i := 0; i < 3; i++ {
			if err := stepOnce(); err != nil {
				fmt.Fprintln(os.Stderr, "warmup:", err)
				os.Exit(1)
			}
			if err := stream.Synchronize(); err != nil {
				fmt.Fprintln(os.Stderr, "warmup sync:", err)
				os.Exit(1)
			}
		}
		graph, err := cuda.Capture(stream, stepOnce)
		if err != nil {
			fmt.Fprintln(os.Stderr, "capture:", err)
			os.Exit(1)
		}
		defer graph.Close()

		var launchTime time.Duration
		wall := time.Now()
		for step := 0; step < *steps; step++ {
			start := time.Now()
			if err := graph.Launch(); err != nil {
				fmt.Fprintln(os.Stderr, "graph launch:", err)
				os.Exit(1)
			}
			launchTime += time.Since(start)
			if step%20 == 0 || step == *steps-1 {
				lossValues, err := gputcn.DownloadFloat32(lossBuffer, 1)
				if err != nil {
					fmt.Fprintln(os.Stderr, "loss download:", err)
					os.Exit(1)
				}
				if step == 0 {
					initial = float64(lossValues[0])
				}
				fmt.Printf("step %3d/%d: loss=%.4f\n", step, *steps, lossValues[0])
			}
		}
		if err := stream.Synchronize(); err != nil {
			fmt.Fprintln(os.Stderr, "final sync:", err)
			os.Exit(1)
		}
		fmt.Printf("initial loss %.4f\n", initial)
		fmt.Printf("average per step: graph launch %.3fms, wall %.3fms\n",
			ms(launchTime, stepsCount), ms(time.Since(wall), stepsCount))
	} else {
		var forwardTime, lossTime, backwardTime, optimTime time.Duration
		for step := 0; step < *steps; step++ {
			start := time.Now()
			cache, err := model.ForwardCached(input, *batch, length)
			if err != nil {
				fmt.Fprintln(os.Stderr, "forward:", err)
				os.Exit(1)
			}
			forwardTime += time.Since(start)

			start = time.Now()
			if err := lossBuffer.Memset(0, 4); err != nil {
				fmt.Fprintln(os.Stderr, "loss reset:", err)
				os.Exit(1)
			}
			if err := kernels.SequenceLossGrad(cache.Output(), targetBuffer, maskBuffer, gradientBuffer, lossBuffer,
				*batch, length, totalValid, totalPairs, options.Bounded, options.LowCents, options.HighCents, options.DeltaWeight); err != nil {
				fmt.Fprintln(os.Stderr, "loss:", err)
				os.Exit(1)
			}
			lossValues, err := gputcn.DownloadFloat32(lossBuffer, 1)
			if err != nil {
				fmt.Fprintln(os.Stderr, "loss download:", err)
				os.Exit(1)
			}
			lossTime += time.Since(start)
			if step == 0 {
				initial = float64(lossValues[0])
			}

			start = time.Now()
			grads, err := model.Backward(cache, gradientBuffer)
			cache.Close()
			if err != nil {
				fmt.Fprintln(os.Stderr, "backward:", err)
				os.Exit(1)
			}
			backwardTime += time.Since(start)

			start = time.Now()
			if err := model.ApplyAdamW(grads, state, *learningRate, 1e-5, 1.0); err != nil {
				fmt.Fprintln(os.Stderr, "optimizer:", err)
				os.Exit(1)
			}
			optimTime += time.Since(start)
			grads.Close()

			if step%20 == 0 || step == *steps-1 {
				fmt.Printf("step %3d/%d: loss=%.4f\n", step, *steps, lossValues[0])
			}
		}
		fmt.Printf("initial loss %.4f\n", initial)
		fmt.Printf("average per step: forward %.3fms, loss %.3fms, backward %.3fms, adamw %.3fms, total %.3fms\n",
			ms(forwardTime, stepsCount), ms(lossTime, stepsCount), ms(backwardTime, stepsCount), ms(optimTime, stepsCount),
			ms(forwardTime+lossTime+backwardTime+optimTime, stepsCount))
	}

	featureNames := make([]string, *inputs)
	for i := range featureNames {
		featureNames[i] = fmt.Sprintf("f%d", i)
	}
	exported, err := model.ExportFramePitch(featureNames, 10, -250, 250, 1)
	if err != nil {
		fmt.Fprintln(os.Stderr, "export:", err)
		os.Exit(1)
	}
	payload, err := json.MarshalIndent(map[string]any{
		"id":               "gputcn-synthetic",
		"version":          8,
		"feature_version":  1,
		"mode":             "intonation_frame_tcn_accent_bounded",
		"language":         "ja",
		"duration_weights": map[string]float64{},
		"frame_pitch":      exported,
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
	fmt.Printf("wrote %s (%d features)\n", *out, len(featureNames))
}

func ms(total time.Duration, steps float64) float64 {
	return float64(total.Microseconds()) / 1000 / math.Max(1, steps)
}

func mustUpload(rng *rand.Rand, count int, scale float64) *cuda.Buffer {
	buffer, err := upload(randomValues(rng, count, scale))
	if err != nil {
		panic(err)
	}
	return buffer
}
