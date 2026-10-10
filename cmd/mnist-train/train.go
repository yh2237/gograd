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
	if ctx == nil || c.Steps < 2 || c.Batch < 1 || c.StopAfter < 0 || c.StopAfter > c.Steps || c.TrainLimit < 0 || c.ValidLimit < 0 || (device != tensor.CPU && device != tensor.CUDA) {
		return stats, fmt.Errorf("invalid training configuration")
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
	loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](trainData, sampler, data.Stack, data.LoaderOptions{BatchSize: c.Batch})
	if err != nil {
		return stats, err
	}
	identity := runIdentity{1, c.Steps, c.Batch, trainData.Len(), validData.Len(), c.Seed, trainHash, validHash}
	schedule := autograd.NewOneCycle(.005, c.Steps, .3)
	optimizer := autograd.NewAdamW(m.module.NamedParameters(), float32(schedule.LR()), .0001)
	defer optimizer.Close()
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
		loss, err := trainingStep(m, execution, optimizer, schedule, batch, device)
		if err != nil {
			return stats, err
		}
		stats.TrainLoss = loss
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
	loss, accuracy, logits, err := evaluate(ctx, m, execution, validData, c.Batch, device)
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
		_, _, restored, err := evaluate(ctx, loaded, execution, validData, c.Batch, device)
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
func trainingStep(m *digitModel, execution *autograd.ExecutionContext, optimizer *autograd.AdamW, schedule *autograd.OneCycle, batch data.FloatBatch, device tensor.Device) (float32, error) {
	x, err := execution.New(batch.Values, batch.Shape, device, false)
	if err != nil {
		return 0, err
	}
	defer x.Close()
	optimizer.ZeroGrad()
	loss := autograd.CrossEntropy(m.forward(x), batch.Labels, -1)
	defer loss.ReleaseGraph()
	values, err := loss.ToHost()
	if err != nil {
		return 0, err
	}
	if err := loss.Backward(); err != nil {
		return 0, err
	}
	optimizer.Step()
	schedule.Step(optimizer)
	return values[0], nil
}
func evaluate(ctx context.Context, m *digitModel, execution *autograd.ExecutionContext, dataset data.Dataset[data.FloatSample], batchSize int, device tensor.Device) (float64, float64, []float32, error) {
	sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{})
	if err != nil {
		return 0, 0, nil, err
	}
	loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](dataset, sampler, data.Stack, data.LoaderOptions{BatchSize: batchSize})
	if err != nil {
		return 0, 0, nil, err
	}
	var total float64
	var correct, count int
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
		total += float64(lossValue[0]) * float64(len(batch.Labels))
		count += len(batch.Labels)
		all = append(all, values...)
		for i, label := range batch.Labels {
			best := 0
			for class := 1; class < 10; class++ {
				if values[i*10+class] > values[i*10+best] {
					best = class
				}
			}
			if best == label {
				correct++
			}
		}
	}
	return total / float64(count), float64(correct) / float64(count), all, nil
}
