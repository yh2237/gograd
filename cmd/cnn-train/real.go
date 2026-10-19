package main

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/tensor"
)

// Keep IDX pixels as bytes; normalize only the samples in the current batch.
type idxRange struct {
	idx          *data.IDX
	start, count int
}

func (d idxRange) Len() int { return d.count }
func (d idxRange) Get(ctx context.Context, index int) (data.FloatSample, error) {
	if index < 0 || index >= d.count {
		return data.FloatSample{}, fmt.Errorf("cnn-train: subset index %d out of range", index)
	}
	return d.idx.Get(ctx, d.start+index)
}

func trainReal(device tensor.Device, steps, trainLimit, validLimit int, imagesPath, labelsPath, output string, log io.Writer, augment bool) (trainingStats, error) {
	stats := trainingStats{}
	if steps < 1 || (device != tensor.CPU && device != tensor.CUDA) {
		return stats, fmt.Errorf("cnn-train: positive steps and cpu/cuda device required")
	}
	if trainLimit < 0 || validLimit < 0 || imagesPath == "" || labelsPath == "" {
		return stats, fmt.Errorf("cnn-train: nonnegative limits and both image/label paths required")
	}
	idx, err := data.OpenIDX(imagesPath, labelsPath)
	if err != nil {
		return stats, err
	}
	h, w := idx.ImageSize()
	if h < 2 || w < 2 {
		return stats, fmt.Errorf("cnn-train: images must be at least 2x2 for pooling")
	}
	nClasses := idx.Classes()
	if nClasses < 2 {
		nClasses = 2
	}

	// Zero limits mean all available samples, without decoding the full subset.
	trainN := idx.Len()
	if trainLimit > 0 && trainLimit < trainN {
		trainN = trainLimit
	}
	validN := idx.Len() - trainN
	if validLimit > 0 && validLimit < validN {
		validN = validLimit
	}
	if trainN <= 0 {
		return stats, fmt.Errorf("cnn-train: need at least one training image")
	}
	baseDS := idxRange{idx: idx, count: trainN}
	validDS := idxRange{idx: idx, start: trainN, count: validN}

	if device == tensor.CUDA {
		ctx, err := autograd.NewCUDAContext()
		if err != nil {
			return stats, err
		}
		defer ctx.Close()
	}
	m, err := newClassifier(device, nClasses, 1)
	if err != nil {
		return stats, err
	}
	defer m.close()
	execution := autograd.NewExecutionContext()
	if err := execution.BindModule(&m.module); err != nil {
		return stats, err
	}
	m.module.Train(true)
	opt, err := newCNNTrainingOptimizer(m.module.NamedParameters(), "bias")
	if err != nil {
		return stats, err
	}
	defer opt.Close()
	trainDS := data.TransformDataset[data.FloatSample]{Base: baseDS, Seed: 7, Apply: func(index int, seed uint64, sample data.FloatSample) (data.FloatSample, error) {
		if !augment {
			return sample, nil
		}
		out := data.FloatSample{Values: append([]float32(nil), sample.Values...), Label: sample.Label, Shape: append([]int(nil), sample.Shape...)}
		// 学習画像だけに変換を適用。横反転・縦反転・ガウスノイズを index/seed に
		// より決定的に適用し、worker 数や読込順に依存させない。
		if err := data.RandomHorizontalFlipFloat32(out.Values, 1, h, w, seed, index, .5); err != nil {
			return data.FloatSample{}, err
		}
		if err := data.RandomVerticalFlipFloat32(out.Values, 1, h, w, seed, index, .3); err != nil {
			return data.FloatSample{}, err
		}
		if err := data.AddGaussianNoiseFloat32(out.Values, seed, index, .02); err != nil {
			return data.FloatSample{}, err
		}
		return out, nil
	}}
	sampler, err := data.NewIndexSampler(trainDS.Len(), data.SamplerOptions{Shuffle: true, Seed: 7})
	if err != nil {
		return stats, err
	}
	loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](trainDS, sampler, data.Stack, data.LoaderOptions{BatchSize: 64})
	if err != nil {
		return stats, err
	}
	defer loader.Close()

	// Initial loss evaluation on a small sample of training data.
	loop := autograd.TrainingLoop[data.FloatBatch]{
		Reader: loader, Sampler: sampler, Optimizer: opt,
		Updates: steps, Applied: 0,
		Step: func(ctx context.Context, batch data.FloatBatch, acc *autograd.GradAccumulator) (float64, bool, error) {
			x, err := execution.New(batch.Values, batch.Shape, device, false)
			if err != nil {
				return 0, false, err
			}
			defer x.Close()
			logits := m.forward(x)
			loss := autograd.CrossEntropy(logits, batch.Labels, -1)
			values, err := loss.ToHost()
			if err != nil {
				loss.ReleaseGraph()
				return 0, false, err
			}
			complete, err := acc.Backward(loss)
			loss.ReleaseGraph()
			if err != nil {
				return 0, false, err
			}
			if stats.initialLoss == 0 && complete {
				stats.initialLoss = values[0]
			}
			return float64(values[0]), complete, nil
		},
		Report: func(update int, loss float64) {
			if update == 1 || update%10 == 0 || update == steps {
				fmt.Fprintf(log, "device=%s step=%d loss=%.6f\n", device, update, loss)
			}
		},
		Save: func([]byte) error {
			if output == "" {
				return nil
			}
			return m.module.SaveSafeTensors(output)
		},
	}
	stopErr := loop.Run(context.Background())
	if stopErr != nil {
		return stats, stopErr
	}

	// With no held-out samples, report actual training-set evaluation rather
	// than substituting initial loss for a measurement that never happened.
	m.module.Train(false)
	evalDS := validDS
	evalSet := "validation"
	if validN == 0 {
		evalDS, evalSet = baseDS, "training"
	}
	stats.finalLoss, stats.accuracy, err = evaluateReal(m, execution, evalDS, device, nClasses)
	if err != nil {
		return stats, err
	}

	if output != "" {
		if err := m.module.SaveSafeTensors(output); err != nil {
			return stats, err
		}
		loaded, err := newClassifier(device, nClasses, 1)
		if err != nil {
			return stats, err
		}
		defer loaded.close()
		if err := execution.BindModule(&loaded.module); err != nil {
			return stats, err
		}
		if err = loaded.module.LoadSafeTensors(output); err != nil {
			return stats, err
		}
		loaded.module.Train(false)
		orig := m.module.StateDict()
		rest := loaded.module.StateDict()
		for name, want := range orig {
			got := rest[name]
			if len(got) != len(want) {
				return stats, fmt.Errorf("cnn-train: checkpoint shape changed for %s", name)
			}
			for i, v := range want {
				if got[i] != v {
					return stats, fmt.Errorf("cnn-train: checkpoint changed %s[%d]", name, i)
				}
			}
		}
		stats.reloadMaxAbs, err = compareRealInference(m, loaded, execution, evalDS, device)
		if err != nil {
			return stats, err
		}
		fmt.Fprintf(log, "saved=%s reload_max_abs=%.9g\n", output, stats.reloadMaxAbs)
	}
	fmt.Fprintf(log, "initial_loss=%.6f final_loss=%.6f accuracy=%.1f%% classes=%d samples=train%d/valid%d evaluation=%s\n",
		stats.initialLoss, stats.finalLoss, 100*stats.accuracy, nClasses, trainN, validN, evalSet)
	return stats, nil
}

func newCNNTrainingOptimizer(params []autograd.Parameter, suffix string) (autograd.CheckpointOptimizer, error) {
	decay := float32(.001)
	if suffix != "" {
		specs, err := autograd.PartitionParameters(params, func(name string) string {
			if strings.HasSuffix(name, suffix) {
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
		return autograd.NewAdamWGroups(params, .01, specs)
	}
	return autograd.NewAdamW(params, .01, .001), nil
}

func realBatch(ds idxRange, start int) (data.FloatBatch, error) {
	if start < 0 || start >= ds.Len() {
		return data.FloatBatch{}, fmt.Errorf("cnn-train: batch start %d out of range", start)
	}
	samples := make([]data.FloatSample, min(64, ds.Len()-start))
	for i := range samples {
		sample, err := ds.Get(context.Background(), start+i)
		if err != nil {
			return data.FloatBatch{}, err
		}
		samples[i] = sample
	}
	return data.Stack(samples)
}

func evaluateReal(m *classifier, execution *autograd.ExecutionContext, ds idxRange, device tensor.Device, classes int) (float32, float32, error) {
	if ds.Len() == 0 {
		return 0, 0, fmt.Errorf("cnn-train: empty evaluation dataset")
	}
	var totalLoss float64
	correct := 0
	for start := 0; start < ds.Len(); start += 64 {
		batch, err := realBatch(ds, start)
		if err != nil {
			return 0, 0, err
		}
		loss, hits, err := evaluateRealBatch(m, execution, batch, device, classes)
		if err != nil {
			return 0, 0, err
		}
		totalLoss += float64(loss) * float64(len(batch.Labels))
		correct += hits
	}
	return float32(totalLoss / float64(ds.Len())), float32(correct) / float32(ds.Len()), nil
}

func evaluateRealBatch(m *classifier, execution *autograd.ExecutionContext, batch data.FloatBatch, device tensor.Device, classes int) (float32, int, error) {
	x, err := execution.New(batch.Values, batch.Shape, device, false)
	if err != nil {
		return 0, 0, err
	}
	defer x.Close()
	var logits, loss *autograd.Tensor
	execution.NoGrad(func() {
		logits = m.forward(x)
		loss = autograd.CrossEntropy(logits, batch.Labels, -1)
	})
	defer loss.ReleaseGraph()
	values, err := logits.ToHost()
	if err != nil {
		return 0, 0, err
	}
	v, err := loss.ToHost()
	if err != nil {
		return 0, 0, err
	}
	if len(v) != 1 || math.IsNaN(float64(v[0])) || math.IsInf(float64(v[0]), 0) {
		return 0, 0, fmt.Errorf("cnn-train: invalid evaluation loss")
	}
	correct := 0
	for i, want := range batch.Labels {
		got := 0
		for c := 1; c < classes; c++ {
			if values[i*classes+c] > values[i*classes+got] {
				got = c
			}
		}
		if got == want {
			correct++
		}
	}
	return v[0], correct, nil
}

func compareRealInference(original, loaded *classifier, execution *autograd.ExecutionContext, ds idxRange, device tensor.Device) (float64, error) {
	batch, err := realBatch(ds, 0)
	if err != nil {
		return 0, err
	}
	x, err := execution.New(batch.Values, batch.Shape, device, false)
	if err != nil {
		return 0, err
	}
	defer x.Close()
	var want, got *autograd.Tensor
	execution.NoGrad(func() { want, got = original.forward(x), loaded.forward(x) })
	defer want.ReleaseGraph()
	defer got.ReleaseGraph()
	w, err := want.ToHost()
	if err != nil {
		return 0, err
	}
	g, err := got.ToHost()
	if err != nil {
		return 0, err
	}
	if len(w) != len(g) {
		return 0, fmt.Errorf("cnn-train: checkpoint changed logits shape")
	}
	var maxAbs float64
	for i := range w {
		diff := math.Abs(float64(w[i]) - float64(g[i]))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			return 0, fmt.Errorf("cnn-train: non-finite checkpoint logits")
		}
		maxAbs = math.Max(maxAbs, diff)
	}
	if maxAbs > 1e-5 {
		return maxAbs, fmt.Errorf("cnn-train: checkpoint changed logits by %g", maxAbs)
	}
	return maxAbs, nil
}
