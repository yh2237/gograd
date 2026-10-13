package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/metrics"
	"github.com/yh2237/gograd/tensor"
)

type digitModel struct {
	module       autograd.Module
	hidden, head *autograd.LinearLayer
}

func newDigitModel(device tensor.Device, seed uint64) (*digitModel, error) {
	m := &digitModel{}
	rng := rand.New(rand.NewSource(int64(seed)))
	var err error
	m.hidden, err = autograd.NewLinearLayer(784, 64, device, rng)
	if err != nil {
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "hidden", Module: m.hidden.StateModule()})
	m.head, err = autograd.NewLinearLayer(64, 10, device, rng)
	if err != nil {
		m.close()
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "head", Module: m.head.StateModule()})
	return m, nil
}
func (m *digitModel) close() {
	for _, p := range m.module.NamedParameters() {
		p.Value.Close()
	}
}
func (m *digitModel) forward(x *autograd.Tensor) *autograd.Tensor {
	return m.head.Forward(autograd.ReLU(m.hidden.Forward(autograd.Reshape(x, x.Shape[0], 784))))
}

type runIdentity struct {
	Version, Steps, Batch, TrainSize, ValidSize int
	Seed                                        uint64
	TrainFingerprint, ValidFingerprint          string
	Optimizer                                   string
	MaxLR, Momentum                             float64
	Nesterov                                    bool
	NoDecaySuffix                               string
	Accum                                       int
}
type checkpointState struct {
	Identity runIdentity
	Loader   data.LoaderState
}
type trainingStats struct {
	Steps                                  int
	TrainLoss                              float32
	ValidationLoss, Accuracy, ReloadMaxAbs float64
}

func readDigits(dir, imagesName, labelsName string, limit int) (*data.Subset[data.FloatSample], string, error) {
	images, err := locateIDX(dir, imagesName)
	if err != nil {
		return nil, "", err
	}
	labels, err := locateIDX(dir, labelsName)
	if err != nil {
		return nil, "", err
	}
	dataset, err := data.OpenIDX(images, labels)
	if err != nil {
		return nil, "", err
	}
	h, w := dataset.ImageSize()
	if h != 28 || w != 28 || dataset.Classes() > 10 {
		return nil, "", fmt.Errorf("MNIST requires 28x28 images and digit labels 0..9")
	}
	n := dataset.Len()
	if limit > 0 {
		if limit > n {
			return nil, "", fmt.Errorf("data limit %d exceeds %d samples", limit, n)
		}
		n = limit
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	subset, err := data.NewSubset[data.FloatSample](dataset, indices)
	return subset, dataset.Fingerprint(), err
}
func train(ctx context.Context, c trainConfig, log io.Writer) (trainingStats, error) {
	var stats trainingStats
	device := tensor.Device(c.Device)
	if c.Optimizer == "" {
		c.Optimizer = "adamw"
	}
	if c.Optimizer != "adamw" && c.Optimizer != "sgd" {
		return stats, fmt.Errorf("unknown optimizer %q", c.Optimizer)
	}
	if c.MaxLR == 0 {
		c.MaxLR = .005
		if c.Optimizer == "sgd" {
			c.MaxLR = .1
		}
	}
	if c.MaxLR <= 0 || math.IsNaN(c.MaxLR) || math.IsInf(c.MaxLR, 0) || c.MaxLR > math.MaxFloat32 || c.Momentum < 0 || math.IsNaN(c.Momentum) || math.IsInf(c.Momentum, 0) || c.Momentum > math.MaxFloat32 {
		return stats, fmt.Errorf("invalid learning rate/momentum")
	}
	if c.Optimizer == "adamw" {
		c.Momentum = 0
		if c.Nesterov {
			return stats, fmt.Errorf("Nesterov applies to SGD only")
		}
	}
	if ctx == nil || c.Steps < 2 || c.Batch < 1 || c.StopAfter < 0 || c.StopAfter > c.Steps || c.TrainLimit < 0 || c.ValidLimit < 0 || c.Workers < 0 || c.Prefetch < 0 || c.Workers == 0 && c.Prefetch != 0 || (device != tensor.CPU && device != tensor.CUDA) {
		return stats, fmt.Errorf("invalid training configuration")
	}
	if c.Accum == 0 {
		c.Accum = 1
	}
	if c.Accum < 1 {
		return stats, fmt.Errorf("accumulation needs at least one micro-batch")
	}
	if c.Checkpoint == "" && c.Out != "" {
		c.Checkpoint = c.Out + ".training.safetensors"
	}
	if c.Out != "" && c.Checkpoint != "" {
		out, err := filepath.Abs(c.Out)
		if err != nil {
			return stats, err
		}
		checkpoint, err := filepath.Abs(c.Checkpoint)
		if err != nil {
			return stats, err
		}
		if out == checkpoint || runtime.GOOS == "windows" && strings.EqualFold(out, checkpoint) {
			return stats, fmt.Errorf("inference and training checkpoint paths must differ")
		}
	}
	if c.Download {
		if err := downloadMNIST(ctx, c.DataDir, mnistURL); err != nil {
			return stats, err
		}
	}
	trainData, trainHash, err := readDigits(c.DataDir, mnistFiles[0], mnistFiles[1], c.TrainLimit)
	if err != nil {
		return stats, err
	}
	validData, validHash, err := readDigits(c.DataDir, mnistFiles[2], mnistFiles[3], c.ValidLimit)
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
	m, err := newDigitModel(device, c.Seed)
	if err != nil {
		return stats, err
	}
	defer m.close()
	execution := autograd.NewExecutionContext()
	if err := execution.BindModule(&m.module); err != nil {
		return stats, err
	}
	sampler, err := data.NewIndexSampler(trainData.Len(), data.SamplerOptions{Shuffle: true, Seed: c.Seed})
	if err != nil {
		return stats, err
	}
	loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](trainData, sampler, data.Stack, data.LoaderOptions{BatchSize: c.Batch, Workers: c.Workers, Prefetch: c.Prefetch})
	if err != nil {
		return stats, err
	}
	defer loader.Close()
	identity := runIdentity{Version: 1, Steps: c.Steps, Batch: c.Batch, TrainSize: trainData.Len(), ValidSize: validData.Len(), Seed: c.Seed, TrainFingerprint: trainHash, ValidFingerprint: validHash, Optimizer: c.Optimizer, MaxLR: c.MaxLR, Momentum: c.Momentum, Nesterov: c.Nesterov, NoDecaySuffix: c.NoDecaySuffix, Accum: c.Accum}
	schedule := autograd.NewOneCycle(c.MaxLR, c.Steps, .3)
	if c.Resume != "" {
		metadata, err := autograd.TrainingCheckpointMetadata(c.Resume)
		if err != nil {
			return stats, err
		}
		var state checkpointState
		if err := json.Unmarshal([]byte(metadata["data_state"]), &state); err != nil {
			return stats, err
		}
		if state.Identity.Version == 1 && state.Identity.Optimizer == "" {
			state.Identity.Optimizer = "adamw"
			state.Identity.MaxLR = .005
			// Older checkpoints predate parameter groups.
			state.Identity.NoDecaySuffix, c.NoDecaySuffix, identity.NoDecaySuffix = "", "", ""
		}
		if state.Identity != identity {
			return stats, fmt.Errorf("checkpoint does not match data/configuration")
		}
		if err := loader.LoadState(state.Loader); err != nil {
			return stats, err
		}
	}
	optimizer, err := newTrainingOptimizer(m.module.NamedParameters(), c, float32(schedule.LR()))
	if err != nil {
		return stats, err
	}
	defer optimizer.Close()
	if c.Resume != "" {
		if _, err := autograd.LoadTrainingCheckpoint(c.Resume, &m.module, optimizer, schedule); err != nil {
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
	m.module.Train(true)
	accumulator, err := autograd.NewGradAccumulator(c.Accum)
	if err != nil {
		return stats, err
	}
	for schedule.StepCount < end {
		batch, err := loader.Next(ctx)
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			break
		}
		if err == io.EOF {
			if err := loader.NextEpoch(); err != nil {
				return stats, err
			}
			continue
		}
		if err != nil {
			return stats, err
		}
		if accumulator.Pending() == 0 {
			optimizer.ZeroGrad()
		}
		loss, complete, err := trainingStep(m, execution, accumulator, batch, device)
		if err != nil {
			return stats, err
		}
		stats.TrainLoss = loss
		if !complete {
			continue
		}
		optimizer.Step()
		schedule.Step(optimizer)
		if schedule.StepCount == 1 || schedule.StepCount%25 == 0 || schedule.StepCount == end {
			fmt.Fprintf(log, "device=%s step=%d train_loss=%.6f epoch=%d\n", device, schedule.StepCount, loss, sampler.State().Epoch)
		}
	}
	stats.Steps = schedule.StepCount
	if c.Checkpoint != "" {
		state, err := loader.State()
		if err != nil {
			return stats, err
		}
		payload, err := json.Marshal(checkpointState{identity, state})
		if err != nil {
			return stats, err
		}
		if err := outputParent(c.Checkpoint); err != nil {
			return stats, err
		}
		if err := autograd.SaveTrainingCheckpoint(c.Checkpoint, &m.module, optimizer, schedule, map[string]string{"data_state": string(payload), "model": "mnist-mlp-1"}); err != nil {
			return stats, err
		}
	}
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	m.module.Train(false)
	loss, accuracy, logits, err := evaluate(ctx, m, execution, validData, c.Batch, c.Workers, c.Prefetch, device)
	if err != nil {
		return stats, err
	}
	stats.ValidationLoss, stats.Accuracy = loss, accuracy
	fmt.Fprintf(log, "completed=%d test_samples=%d test_loss=%.6f test_accuracy=%.4f\n", stats.Steps, validData.Len(), loss, accuracy)
	if c.Out != "" {
		if err := outputParent(c.Out); err != nil {
			return stats, err
		}
		if err := m.module.SaveSafeTensors(c.Out); err != nil {
			return stats, err
		}
		loaded, err := newDigitModel(device, c.Seed)
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
		for i, v := range logits {
			stats.ReloadMaxAbs = math.Max(stats.ReloadMaxAbs, math.Abs(float64(v-restored[i])))
		}
		if stats.ReloadMaxAbs > 1e-5 {
			return stats, fmt.Errorf("inference reload changed logits by %g", stats.ReloadMaxAbs)
		}
		fmt.Fprintf(log, "saved=%s checkpoint=%s reload_max_abs=%.9g\n", c.Out, c.Checkpoint, stats.ReloadMaxAbs)
	}
	return stats, nil
}

// trainingStep builds one micro-batch loss and adds its scaled gradient to the
// accumulator. It reports whether the call completed an optimizer update; the
// caller owns ZeroGrad, Step and the scheduler.
func trainingStep(m *digitModel, execution *autograd.ExecutionContext, accumulator *autograd.GradAccumulator, batch data.FloatBatch, device tensor.Device) (float32, bool, error) {
	x, err := execution.New(batch.Values, batch.Shape, device, false)
	if err != nil {
		return 0, false, err
	}
	defer x.Close()
	loss := autograd.CrossEntropy(m.forward(x), batch.Labels, -1)
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
	return values[0], complete, nil
}

func newTrainingOptimizer(params []autograd.Parameter, c trainConfig, lr float32) (autograd.CheckpointOptimizer, error) {
	decay := float32(.0001)
	options := autograd.SGDOptions{LR: lr, Momentum: float32(c.Momentum), Nesterov: c.Nesterov}
	if c.NoDecaySuffix != "" {
		// Biases and similar parameters get their own group without decay, which
		// is the common reason to need groups in the first place.
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
		return autograd.NewSGD(params, autograd.SGDOptions{LR: lr, Momentum: float32(c.Momentum), Nesterov: c.Nesterov, WeightDecay: decay})
	}
	return autograd.NewAdamW(params, lr, decay), nil
}
func evaluate(ctx context.Context, m *digitModel, execution *autograd.ExecutionContext, dataset data.Dataset[data.FloatSample], batchSize, workers, prefetch int, device tensor.Device) (float64, float64, []float32, error) {
	sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{})
	if err != nil {
		return 0, 0, nil, err
	}
	loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](dataset, sampler, data.Stack, data.LoaderOptions{BatchSize: batchSize, Workers: workers, Prefetch: prefetch})
	if err != nil {
		return 0, 0, nil, err
	}
	defer loader.Close()
	var mean metrics.Mean
	var accuracy metrics.Accuracy
	var all []float32
	for {
		batch, err := loader.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, 0, nil, err
		}
		x, err := execution.New(batch.Values, batch.Shape, device, false)
		if err != nil {
			return 0, 0, nil, err
		}
		var prediction, loss *autograd.Tensor
		execution.NoGrad(func() { prediction = m.forward(x); loss = autograd.CrossEntropy(prediction, batch.Labels, -1) })
		values, err := prediction.ToHost()
		if err != nil {
			loss.ReleaseGraph()
			x.Close()
			return 0, 0, nil, err
		}
		lossValue, err := loss.ToHost()
		loss.ReleaseGraph()
		x.Close()
		if err != nil {
			return 0, 0, nil, err
		}
		if err := mean.Add(float64(lossValue[0]), len(batch.Labels)); err != nil {
			return 0, 0, nil, err
		}
		if err := accuracy.UpdateLogits(values, batch.Labels, 10, -1); err != nil {
			return 0, 0, nil, err
		}
		all = append(all, values...)
	}
	if mean.Count() == 0 {
		return 0, 0, nil, fmt.Errorf("no validation samples")
	}
	return mean.Value(), accuracy.Value(), all, nil
}
