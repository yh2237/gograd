package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	randv2 "math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/metrics"
	"github.com/yh2237/gograd/tensor"
)

type regressionModel struct {
	module       autograd.Module
	hidden, head *autograd.LinearLayer
}

func newRegressionModel(features, hidden int, device tensor.Device, seed uint64) (*regressionModel, error) {
	m := &regressionModel{}
	rng := rand.New(rand.NewSource(int64(seed)))
	var err error
	m.hidden, err = autograd.NewLinearLayer(features, hidden, device, rng)
	if err != nil {
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "hidden", Module: m.hidden.StateModule()})
	m.head, err = autograd.NewLinearLayer(hidden, 1, device, rng)
	if err != nil {
		m.close()
		return nil, err
	}
	m.module.Children = append(m.module.Children, autograd.NamedModule{Name: "head", Module: m.head.StateModule()})
	return m, nil
}
func (m *regressionModel) close() {
	for _, p := range m.module.NamedParameters() {
		p.Value.Close()
	}
}
func (m *regressionModel) forward(x *autograd.Tensor) *autograd.Tensor {
	return m.head.Forward(autograd.ReLU(m.hidden.Forward(x)))
}

type runIdentity struct {
	Version, Steps, Batch, Hidden, Rows, Features, ValidRows int
	Seed                                                     uint64
	Fingerprint                                              string
	Optimizer                                                string
	MaxLR, Momentum                                          float64
}
type checkpointState struct {
	Identity     runIdentity
	Loader       data.LoaderState
	Standardizer data.StandardizerState
}
type trainingStats struct {
	Steps                int
	TrainLoss, ValidRMSE float64
	ReloadMaxAbs         float64
}

// loadTable reads the CSV once so its fingerprint and parsed rows describe the
// same bytes. Feature standardization is applied after the split, using training
// statistics only.
func loadTable(path string) ([]data.RegressionSample, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	table, err := data.ParseNumericCSV(bytes.NewReader(raw), data.ParseOptions{})
	if err != nil {
		return nil, "", err
	}
	samples, err := table.RegressionSamples(nil, -1)
	if err != nil {
		return nil, "", err
	}
	fingerprint := fmt.Sprintf("%x", sha256.Sum256(raw))
	return samples, fingerprint, nil
}

func standardizeAll(samples []data.RegressionSample, standardizer data.Standardizer) []data.RegressionSample {
	out := make([]data.RegressionSample, len(samples))
	for i, sample := range samples {
		out[i] = standardizer.Apply(sample)
	}
	return out
}

func generateDataset(path string, rows, features int, seed uint64) error {
	if rows < 2 || features < 1 {
		return fmt.Errorf("invalid generated dataset shape")
	}
	coeff := []float64{0.4, 250, 1.5, 0.7}
	scales := []float64{100, 0.01, 1, 3}
	rng := randv2.New(randv2.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	var builder strings.Builder
	for j := 0; j < features; j++ {
		fmt.Fprintf(&builder, "f%d,", j)
	}
	builder.WriteString("target\n")
	for i := 0; i < rows; i++ {
		values := make([]float64, features)
		var target float64
		for j := range values {
			value := rng.NormFloat64() * scales[j%len(scales)]
			values[j] = value
			switch j % 4 {
			case 0:
				target += coeff[0] * value
			case 1:
				target += coeff[1] * value * value
			case 2:
				target += coeff[2] * math.Sin(value)
			default:
				target += coeff[3] * value
			}
		}
		target += rng.NormFloat64() * 0.2
		for _, value := range values {
			builder.WriteString(strconv.FormatFloat(value, 'g', -1, 32))
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.FormatFloat(target, 'g', -1, 32))
		builder.WriteByte('\n')
	}
	if err := outputParent(path); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(builder.String()), 0600)
}

func outputParent(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0755)
}

func validateConfig(c *trainConfig) error {
	if c.Data == "" || c.Out == "" {
		return fmt.Errorf("data and out paths are required")
	}
	if c.Optimizer == "" {
		c.Optimizer = "adamw"
	}
	if c.Optimizer != "adamw" && c.Optimizer != "sgd" {
		return fmt.Errorf("unknown optimizer %q", c.Optimizer)
	}
	if c.LearningRate == 0 {
		c.LearningRate = .01
		if c.Optimizer == "sgd" {
			c.LearningRate = .05
		}
	}
	if c.LearningRate <= 0 || math.IsNaN(c.LearningRate) || math.IsInf(c.LearningRate, 0) || c.LearningRate > math.MaxFloat32 ||
		c.Momentum < 0 || math.IsNaN(c.Momentum) || math.IsInf(c.Momentum, 0) || c.Momentum > math.MaxFloat32 {
		return fmt.Errorf("invalid learning rate or momentum")
	}
	if c.Optimizer == "adamw" {
		c.Momentum = 0
	}
	if c.Hidden < 1 || c.Steps < 2 || c.Batch < 1 || c.StopAfter < 0 || c.StopAfter > c.Steps || c.ValidCount < 0 ||
		c.Workers < 0 || c.Prefetch < 0 || c.Workers == 0 && c.Prefetch != 0 {
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

func newTrainingOptimizer(params []autograd.Parameter, c trainConfig, lr float32) (autograd.CheckpointOptimizer, error) {
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
		if err := generateDataset(c.Data, c.Rows, c.Features, c.Seed); err != nil {
			return stats, err
		}
		fmt.Fprintf(log, "generated=%s rows=%d features=%d\n", c.Data, c.Rows, c.Features)
	}
	samples, fingerprint, err := loadTable(c.Data)
	if err != nil {
		return stats, err
	}
	if len(samples) < 2 {
		return stats, fmt.Errorf("need at least two rows")
	}
	features := len(samples[0].Features)
	validCount := c.ValidCount
	if validCount == 0 {
		validCount = max(1, len(samples)/5)
	}
	if validCount >= len(samples) {
		return stats, fmt.Errorf("valid-count %d leaves no training rows", validCount)
	}
	trainIndices, validIndices, err := data.RandomSplit(len(samples), validCount, c.Seed)
	if err != nil {
		return stats, err
	}
	trainSamples := make([]data.RegressionSample, len(trainIndices))
	for i, index := range trainIndices {
		trainSamples[i] = samples[index]
	}
	standardizer, err := data.FitStandardizer(trainSamples)
	if err != nil {
		return stats, err
	}
	samples = standardizeAll(samples, standardizer)
	trainData, err := data.NewSubset[data.RegressionSample](data.SliceDataset[data.RegressionSample](samples), trainIndices)
	if err != nil {
		return stats, err
	}
	validData, err := data.NewSubset[data.RegressionSample](data.SliceDataset[data.RegressionSample](samples), validIndices)
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
	model, err := newRegressionModel(features, c.Hidden, device, c.Seed)
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
	loader, err := data.NewLoader[data.RegressionSample, data.RegressionBatch](trainData, sampler, data.StackRegression, data.LoaderOptions{BatchSize: c.Batch, Workers: c.Workers, Prefetch: c.Prefetch})
	if err != nil {
		return stats, err
	}
	defer loader.Close()
	identity := runIdentity{Version: 1, Steps: c.Steps, Batch: c.Batch, Hidden: c.Hidden, Rows: len(samples), Features: features,
		ValidRows: validCount, Seed: c.Seed, Fingerprint: fingerprint, Optimizer: c.Optimizer, MaxLR: c.LearningRate, Momentum: c.Momentum}
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
		// The transform is recomputed deterministically from the training split;
		// it must match what the checkpoint stored.
		if !reflect.DeepEqual(state.Standardizer, standardizer.State()) {
			return stats, fmt.Errorf("checkpoint standardizer does not match the training split")
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
		loss, err := trainingStep(model, execution, optimizer, schedule, batch, device)
		if err != nil {
			return stats, err
		}
		stats.TrainLoss = float64(loss)
		if schedule.StepCount == 1 || schedule.StepCount%25 == 0 || schedule.StepCount == end {
			fmt.Fprintf(log, "device=%s step=%d train_mse=%.6f epoch=%d\n", device, schedule.StepCount, loss, sampler.State().Epoch)
		}
	}
	stats.Steps = schedule.StepCount
	if c.Checkpoint != "" {
		loaderState, err := loader.State()
		if err != nil {
			return stats, err
		}
		payload, err := json.Marshal(checkpointState{Identity: identity, Loader: loaderState, Standardizer: standardizer.State()})
		if err != nil {
			return stats, err
		}
		if err := outputParent(c.Checkpoint); err != nil {
			return stats, err
		}
		if err := autograd.SaveTrainingCheckpoint(c.Checkpoint, &model.module, optimizer, schedule, map[string]string{"data_state": string(payload), "model": "tabular-regression-1"}); err != nil {
			return stats, err
		}
	}
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	model.module.Train(false)
	validRMSE, predictions, err := evaluate(ctx, model, execution, validData, c.Batch, c.Workers, c.Prefetch, device)
	if err != nil {
		return stats, err
	}
	stats.ValidRMSE = validRMSE
	fmt.Fprintf(log, "completed=%d validation_rows=%d validation_rmse=%.6f\n", stats.Steps, validData.Len(), validRMSE)
	if c.Out != "" {
		if err := outputParent(c.Out); err != nil {
			return stats, err
		}
		if err := model.module.SaveSafeTensors(c.Out); err != nil {
			return stats, err
		}
		loaded, err := newRegressionModel(features, c.Hidden, device, c.Seed)
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
		_, restored, err := evaluate(ctx, loaded, execution, validData, c.Batch, c.Workers, c.Prefetch, device)
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

func trainingStep(m *regressionModel, execution *autograd.ExecutionContext, optimizer autograd.Optimizer, schedule *autograd.OneCycle, batch data.RegressionBatch, device tensor.Device) (float32, error) {
	x, err := execution.New(batch.Values, batch.Shape, device, false)
	if err != nil {
		return 0, err
	}
	defer x.Close()
	target, err := execution.New(batch.Targets, []int{batch.Shape[0], 1}, device, false)
	if err != nil {
		return 0, err
	}
	defer target.Close()
	optimizer.ZeroGrad()
	loss := autograd.MSELoss(m.forward(x), target, autograd.LossOptions{})
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

// evaluate returns the validation RMSE and every prediction, so training can
// compare them with a reloaded inference model.
func evaluate(ctx context.Context, m *regressionModel, execution *autograd.ExecutionContext, dataset data.Dataset[data.RegressionSample], batchSize, workers, prefetch int, device tensor.Device) (float64, []float32, error) {
	sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{})
	if err != nil {
		return 0, nil, err
	}
	loader, err := data.NewLoader[data.RegressionSample, data.RegressionBatch](dataset, sampler, data.StackRegression, data.LoaderOptions{BatchSize: batchSize, Workers: workers, Prefetch: prefetch})
	if err != nil {
		return 0, nil, err
	}
	defer loader.Close()
	var regression metrics.Regression
	predictions := make([]float32, 0, dataset.Len())
	for {
		batch, err := loader.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, nil, err
		}
		x, err := execution.New(batch.Values, batch.Shape, device, false)
		if err != nil {
			return 0, nil, err
		}
		var prediction *autograd.Tensor
		execution.NoGrad(func() { prediction = m.forward(x) })
		values, err := prediction.ToHost()
		prediction.ReleaseGraph()
		x.Close()
		if err != nil {
			return 0, nil, err
		}
		if err := regression.Update(values, batch.Targets); err != nil {
			return 0, nil, err
		}
		predictions = append(predictions, values...)
	}
	if len(predictions) == 0 {
		return 0, nil, fmt.Errorf("no validation rows")
	}
	return regression.RMSE(), predictions, nil
}
