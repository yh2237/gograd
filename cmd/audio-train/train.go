package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	randv2 "math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/tensor"
)

type denoiser struct {
	module       autograd.Module
	hidden, head *autograd.LinearLayer
}

func newDenoiser(window, hidden int, device tensor.Device, seed uint64) (*denoiser, error) {
	m := &denoiser{}
	rng := rand.New(rand.NewSource(int64(seed)))
	var err error
	m.hidden, err = autograd.NewLinearLayer(window, hidden, device, rng)
	if err != nil {
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "hidden", Module: m.hidden.StateModule()})
	m.head, err = autograd.NewLinearLayer(hidden, window, device, rng)
	if err != nil {
		m.close()
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "head", Module: m.head.StateModule()})
	return m, nil
}
func (m *denoiser) close() {
	for _, p := range m.module.NamedParameters() {
		p.Value.Close()
	}
}
func (m *denoiser) forward(x *autograd.Tensor) *autograd.Tensor {
	return m.head.Forward(autograd.ReLU(m.hidden.Forward(x)))
}

// windowSample is an application-defined sample type: the generic Loader only
// needs Len/Get and a collate function.
type windowSample struct {
	Input, Target []float32
}
type windowBatch struct {
	Inputs, Targets []float32
	Shape           []int
}

func collateWindows(samples []windowSample) (windowBatch, error) {
	if len(samples) == 0 {
		return windowBatch{}, fmt.Errorf("audio: cannot collate an empty batch")
	}
	width := len(samples[0].Target)
	if width == 0 {
		return windowBatch{}, fmt.Errorf("audio: empty window")
	}
	batch := windowBatch{Inputs: make([]float32, len(samples)*width), Targets: make([]float32, len(samples)*width), Shape: []int{len(samples), width}}
	for i, sample := range samples {
		if len(sample.Target) != width || len(sample.Input) != width {
			return windowBatch{}, fmt.Errorf("audio: collate requires equal window lengths")
		}
		copy(batch.Inputs[i*width:], sample.Input)
		copy(batch.Targets[i*width:], sample.Target)
	}
	return batch, nil
}

// noisyWindows augments a fixed set of clean frames with deterministic noise.
// The noise is derived from the sample index and the seed, never from iteration
// order, so worker/prefetch changes and resumes reproduce the same inputs. The
// seed is checkpoint identity; the frames themselves are the unmodified target.
type noisyWindows struct {
	clean [][]float32
	noise float32
	seed  uint64
}

func (d *noisyWindows) Len() int { return len(d.clean) }
func (d *noisyWindows) Get(ctx context.Context, index int) (windowSample, error) {
	var zero windowSample
	if ctx == nil {
		return zero, fmt.Errorf("audio: nil context")
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if index < 0 || index >= len(d.clean) {
		return zero, fmt.Errorf("audio: window index %d out of range", index)
	}
	target := d.clean[index]
	input := append([]float32(nil), target...)
	if err := data.AddGaussianNoiseFloat32(input, d.seed, index, d.noise); err != nil {
		return zero, err
	}
	return windowSample{Input: input, Target: append([]float32(nil), target...)}, nil
}

// frames splits a mono signal into non-overlapping fixed windows and drops a
// trailing partial window so every sample has the same shape.
func frames(signal []float32, window int) [][]float32 {
	count := len(signal) / window
	out := make([][]float32, count)
	for i := range out {
		out[i] = append([]float32(nil), signal[i*window:(i+1)*window]...)
	}
	return out
}

func generateSignal(samples, sampleRate int, seed uint64) []float32 {
	rng := randv2.New(randv2.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	partials := make([]float64, 3)
	phases := make([]float64, 3)
	for i := range partials {
		partials[i] = 120 * math.Pow(6, float64(i)) // 120, 720, 4320 Hz
		phases[i] = rng.Float64() * 2 * math.Pi
	}
	out := make([]float32, samples)
	for i := range out {
		t := float64(i) / float64(sampleRate)
		var value float64
		for j, frequency := range partials {
			value += math.Sin(2*math.Pi*frequency*t+phases[j]) / float64(j+1)
		}
		// A slow envelope makes the task less trivial than a fixed tone.
		value *= 0.5 + 0.3*math.Sin(2*math.Pi*1.5*t)
		out[i] = float32(value) * 0.6
	}
	return out
}

type runIdentity struct {
	Version, Steps, Batch, Hidden, Window, Frames, ValidCount int
	SampleRate                                                int
	Seed                                                      uint64
	Fingerprint                                               string
	Optimizer                                                 string
	Noise, MaxLR, Momentum                                    float64
}
type checkpointState struct {
	Identity runIdentity
	Loader   data.LoaderState
}
type trainingStats struct {
	Steps                         int
	TrainMSE, InputSNR, OutputSNR float64
	ReloadMaxAbs                  float64
}

func validateConfig(c *trainConfig) error {
	if c.WAV == "" || c.Out == "" {
		return fmt.Errorf("wav and out paths are required")
	}
	if c.Optimizer == "" {
		c.Optimizer = "adamw"
	}
	if c.Optimizer != "adamw" && c.Optimizer != "sgd" {
		return fmt.Errorf("unknown optimizer %q", c.Optimizer)
	}
	if c.Optimizer == "adamw" {
		c.Momentum = 0
	}
	if c.LearningRate == 0 {
		c.LearningRate = .005
		if c.Optimizer == "sgd" {
			c.LearningRate = .02
		}
	}
	for name, value := range map[string]float64{"learning rate": c.LearningRate, "noise": c.Noise, "momentum": c.Momentum} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > math.MaxFloat32 {
			return fmt.Errorf("invalid %s", name)
		}
	}
	if c.LearningRate == 0 {
		return fmt.Errorf("learning rate must be positive")
	}
	if c.Window < 1 || c.Hidden < 1 || c.Steps < 2 || c.Batch < 1 || c.StopAfter < 0 || c.StopAfter > c.Steps ||
		c.ValidCount < 0 || c.Workers < 0 || c.Prefetch < 0 || c.Workers == 0 && c.Prefetch != 0 {
		return fmt.Errorf("invalid training configuration")
	}
	if c.Checkpoint == "" {
		c.Checkpoint = c.Out + ".training.safetensors"
	}
	if c.Out == c.Checkpoint {
		return fmt.Errorf("inference and training checkpoint paths must differ")
	}
	return nil
}

func outputParent(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0755)
}

func writeGeneratedWAV(path string, samples, sampleRate int, seed uint64) error {
	if samples < 1 || sampleRate <= 0 {
		return fmt.Errorf("invalid generated audio shape")
	}
	signal := generateSignal(samples, sampleRate, seed)
	if err := outputParent(path); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return data.WriteWAV(file, signal, data.WAVEncoding{Channels: 1, SampleRate: sampleRate, Bits: 16})
}

func newTrainingOptimizer(params []autograd.Parameter, c trainConfig, lr float32) (autograd.CheckpointOptimizer, error) {
	decay := float32(.0001)
	options := autograd.SGDOptions{LR: lr, Momentum: float32(c.Momentum)}
	if c.NoDecaySuffix != "" {
		specs, err := autograd.PartitionParameters(params, func(name string) string {
			if strings.HasSuffix(name, c.NoDecaySuffix) {
				return "no_decay"
			}
			return "decay"
		})
		if err != nil {
			return nil, err
		}
		for i := range specs {
			if specs[i].Name == "no_decay" {
				specs[i].WeightDecay = 0
			} else {
				specs[i].WeightDecay = decay
			}
		}
		if c.Optimizer == "sgd" {
			return autograd.NewSGDGroups(params, options, specs)
		}
		return autograd.NewAdamWGroups(params, lr, specs)
	}
	if c.Optimizer == "sgd" {
		return autograd.NewSGD(params, autograd.SGDOptions{LR: lr, Momentum: float32(c.Momentum)})
	}
	return autograd.NewAdamW(params, lr, .0001), nil
}

func train(ctx context.Context, c trainConfig, log io.Writer) (trainingStats, error) {
	var stats trainingStats
	if err := validateConfig(&c); err != nil {
		return stats, err
	}
	if ctx == nil {
		return stats, fmt.Errorf("nil context")
	}
	device := tensor.Device(c.Device)
	if device != tensor.CPU && device != tensor.CUDA {
		return stats, fmt.Errorf("unknown device %q", c.Device)
	}
	if c.Generate {
		samples := int(c.Seconds * float64(c.SampleRate))
		if err := writeGeneratedWAV(c.WAV, samples, c.SampleRate, c.Seed); err != nil {
			return stats, err
		}
		fmt.Fprintf(log, "generated=%s samples=%d sample_rate=%d\n", c.WAV, samples, c.SampleRate)
	}
	raw, err := os.ReadFile(c.WAV)
	if err != nil {
		return stats, err
	}
	wav, err := data.ReadWAV(bytes.NewReader(raw))
	if err != nil {
		return stats, err
	}
	signal := wav.Mono()
	clean := frames(signal, c.Window)
	if len(clean) < 2 {
		return stats, fmt.Errorf("WAV has fewer than two windows of %d samples", c.Window)
	}
	validCount := c.ValidCount
	if validCount == 0 {
		validCount = max(1, len(clean)/5)
	}
	if validCount >= len(clean) {
		return stats, fmt.Errorf("valid-count %d leaves no training windows", validCount)
	}
	trainIndices, validIndices, err := data.RandomSplit(len(clean), validCount, c.Seed)
	if err != nil {
		return stats, err
	}
	dataset := &noisyWindows{clean: clean, noise: float32(c.Noise), seed: c.Seed}
	trainData, err := data.NewSubset[windowSample](dataset, trainIndices)
	if err != nil {
		return stats, err
	}
	validData, err := data.NewSubset[windowSample](dataset, validIndices)
	if err != nil {
		return stats, err
	}
	if device == tensor.CUDA {
		gpu, err := autograd.NewCUDAContext()
		if err != nil {
			return stats, err
		}
		defer gpu.Close()
	}
	model, err := newDenoiser(c.Window, c.Hidden, device, c.Seed)
	if err != nil {
		return stats, err
	}
	defer model.close()
	execution := autograd.NewExecutionContext()
	if err := execution.BindModule(&model.module); err != nil {
		return stats, err
	}
	sampler, err := data.NewIndexSampler(trainData.Len(), data.SamplerOptions{Shuffle: true, Seed: c.Seed})
	if err != nil {
		return stats, err
	}
	loader, err := data.NewLoader[windowSample, windowBatch](trainData, sampler, collateWindows, data.LoaderOptions{BatchSize: c.Batch, Workers: c.Workers, Prefetch: c.Prefetch})
	if err != nil {
		return stats, err
	}
	defer loader.Close()
	identity := runIdentity{Version: 1, Steps: c.Steps, Batch: c.Batch, Hidden: c.Hidden, Window: c.Window,
		Frames: len(clean), ValidCount: validCount, SampleRate: wav.SampleRate, Seed: c.Seed,
		Fingerprint: fmt.Sprintf("%x", sha256.Sum256(raw)), Optimizer: c.Optimizer, Noise: c.Noise,
		MaxLR: c.LearningRate, Momentum: c.Momentum}
	schedule := autograd.NewOneCycle(c.LearningRate, c.Steps, .3)
	if c.Resume != "" {
		metadata, err := autograd.TrainingCheckpointMetadata(c.Resume)
		if err != nil {
			return stats, err
		}
		var state checkpointState
		if err := json.Unmarshal([]byte(metadata["data_state"]), &state); err != nil {
			return stats, err
		}
		if state.Identity != identity {
			return stats, fmt.Errorf("checkpoint does not match data/configuration")
		}
		if err := loader.LoadState(state.Loader); err != nil {
			return stats, err
		}
	}
	optimizer, err := newTrainingOptimizer(model.module.NamedParameters(), c, float32(schedule.LR()))
	if err != nil {
		return stats, err
	}
	defer optimizer.Close()
	if c.Resume != "" {
		if _, err := autograd.LoadTrainingCheckpoint(c.Resume, &model.module, optimizer, schedule); err != nil {
			return stats, err
		}
		if schedule.Total != c.Steps {
			return stats, fmt.Errorf("checkpoint schedule does not match planned updates")
		}
	}
	end := c.Steps
	if c.StopAfter > 0 {
		end = c.StopAfter
	}
	model.module.Train(true)
	loop := autograd.TrainingLoop[windowBatch]{
		Reader: loader, Sampler: sampler, Optimizer: optimizer, Schedule: schedule,
		Updates: end, Applied: schedule.StepCount,
		Step: trainingStep(model, execution, device),
		Report: func(update int, loss float64) {
			stats.TrainMSE = loss
			fmt.Fprintf(log, "device=%s step=%d train_mse=%.6f epoch=%d\n", device, update, loss, sampler.State().Epoch)
		},
		Save: func([]byte) error {
			loaderState, err := loader.State()
			if err != nil {
				return err
			}
			payload, err := json.Marshal(checkpointState{Identity: identity, Loader: loaderState})
			if err != nil {
				return err
			}
			if err := outputParent(c.Checkpoint); err != nil {
				return err
			}
			return autograd.SaveTrainingCheckpoint(c.Checkpoint, &model.module, optimizer, schedule, map[string]string{"data_state": string(payload), "model": "audio-denoise-1"})
		},
	}
	// The loop saves the checkpoint of the last completed update, including when
	// the context ends the run.
	stopErr := loop.Run(ctx)
	stats.Steps = schedule.StepCount
	if stopErr != nil {
		return stats, stopErr
	}
	model.module.Train(false)
	inputSNR, outputSNR, predictions, err := evaluate(ctx, model, execution, validData, c.Batch, c.Workers, c.Prefetch, device)
	if err != nil {
		return stats, err
	}
	stats.InputSNR, stats.OutputSNR = inputSNR, outputSNR
	fmt.Fprintf(log, "completed=%d validation_windows=%d input_snr_db=%.4f output_snr_db=%.4f\n", stats.Steps, validData.Len(), inputSNR, outputSNR)
	if c.Out != "" {
		if err := outputParent(c.Out); err != nil {
			return stats, err
		}
		if err := model.module.SaveSafeTensors(c.Out); err != nil {
			return stats, err
		}
		loaded, err := newDenoiser(c.Window, c.Hidden, device, c.Seed)
		if err != nil {
			return stats, err
		}
		defer loaded.close()
		if err := execution.BindModule(&loaded.module); err != nil {
			return stats, err
		}
		if err := loaded.module.LoadSafeTensors(c.Out); err != nil {
			return stats, err
		}
		loaded.module.Train(false)
		_, _, restored, err := evaluate(ctx, loaded, execution, validData, c.Batch, c.Workers, c.Prefetch, device)
		if err != nil {
			return stats, err
		}
		for i, value := range predictions {
			stats.ReloadMaxAbs = math.Max(stats.ReloadMaxAbs, math.Abs(float64(value-restored[i])))
		}
		if stats.ReloadMaxAbs > 1e-5 {
			return stats, fmt.Errorf("inference reload changed predictions by %g", stats.ReloadMaxAbs)
		}
		fmt.Fprintf(log, "saved=%s checkpoint=%s reload_max_abs=%.9g\n", c.Out, c.Checkpoint, stats.ReloadMaxAbs)
	}
	return stats, nil
}

// trainingStep returns the loop's step for the denoiser: it builds the graph,
// back-propagates the loss through the loop's accumulator and reports the loss
// value with the update completion flag. It owns the tensors it creates.
func trainingStep(m *denoiser, execution *autograd.ExecutionContext, device tensor.Device) func(context.Context, windowBatch, *autograd.GradAccumulator) (float64, bool, error) {
	return func(ctx context.Context, batch windowBatch, accumulator *autograd.GradAccumulator) (float64, bool, error) {
		if err := ctx.Err(); err != nil {
			return 0, false, err
		}
		x, err := execution.New(batch.Inputs, batch.Shape, device, false)
		if err != nil {
			return 0, false, err
		}
		defer x.Close()
		target, err := execution.New(batch.Targets, batch.Shape, device, false)
		if err != nil {
			return 0, false, err
		}
		defer target.Close()
		loss := autograd.MSELoss(m.forward(x), target, autograd.LossOptions{})
		values, err := loss.ToHost()
		if err != nil {
			loss.ReleaseGraph()
			return 0, false, err
		}
		complete, err := accumulator.Backward(loss)
		loss.ReleaseGraph()
		if err != nil {
			return 0, false, err
		}
		return float64(values[0]), complete, nil
	}
}

// evaluate returns the noisy input and reconstructed output SNR in dB plus every
// prediction, so training can compare them with a reloaded inference model.
func evaluate(ctx context.Context, m *denoiser, execution *autograd.ExecutionContext, dataset data.Dataset[windowSample], batchSize, workers, prefetch int, device tensor.Device) (float64, float64, []float32, error) {
	sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{})
	if err != nil {
		return 0, 0, nil, err
	}
	loader, err := data.NewLoader[windowSample, windowBatch](dataset, sampler, collateWindows, data.LoaderOptions{BatchSize: batchSize, Workers: workers, Prefetch: prefetch})
	if err != nil {
		return 0, 0, nil, err
	}
	defer loader.Close()
	var signalPower, inputError, outputError float64
	predictions := make([]float32, 0, dataset.Len()*1)
	for {
		batch, err := loader.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, 0, nil, err
		}
		x, err := execution.New(batch.Inputs, batch.Shape, device, false)
		if err != nil {
			return 0, 0, nil, err
		}
		var prediction *autograd.Tensor
		execution.NoGrad(func() { prediction = m.forward(x) })
		values, err := prediction.ToHost()
		prediction.ReleaseGraph()
		x.Close()
		if err != nil {
			return 0, 0, nil, err
		}
		for i, value := range values {
			target := float64(batch.Targets[i])
			signalPower += target * target
			inputError += float64(batch.Inputs[i]-batch.Targets[i]) * float64(batch.Inputs[i]-batch.Targets[i])
			difference := float64(value) - target
			outputError += difference * difference
		}
		predictions = append(predictions, values...)
	}
	if len(predictions) == 0 {
		return 0, 0, nil, fmt.Errorf("no validation windows")
	}
	if signalPower == 0 {
		return 0, 0, predictions, fmt.Errorf("silent validation signal")
	}
	snr := func(noise float64) float64 {
		if noise == 0 {
			return math.Inf(1)
		}
		return 10 * math.Log10(signalPower/noise)
	}
	return snr(inputError), snr(outputError), predictions, nil
}
