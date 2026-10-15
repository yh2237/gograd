package main

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/tensor"
)

type sliceDataset struct{ samples []data.FloatSample }

func (d sliceDataset) Len() int { return len(d.samples) }
func (d sliceDataset) Get(ctx context.Context, index int) (data.FloatSample, error) {
	return d.samples[index], nil
}

func trainReal(device tensor.Device, steps, trainLimit, validLimit int, imagesPath, labelsPath, output string, log io.Writer, augment bool) (trainingStats, error) {
	stats := trainingStats{}
	if steps < 1 || (device != tensor.CPU && device != tensor.CUDA) {
		return stats, fmt.Errorf("cnn-train: positive steps and cpu/cuda device required")
	}
	if device == tensor.CUDA {
		ctx, err := autograd.NewCUDAContext()
		if err != nil {
			return stats, err
		}
		defer ctx.Close()
	}
	idx, err := data.OpenIDX(imagesPath, labelsPath)
	if err != nil {
		return stats, err
	}
	h, w := idx.ImageSize()
	nClasses := idx.Classes()
	if nClasses < 2 {
		nClasses = 2
	}

	// Build train/validation sample slices.
	trainN := idx.Len()
	if trainLimit > 0 && trainLimit < trainN {
		trainN = trainLimit
	}
	validN := trainLimit
	if idx.Len()-trainN < validN {
		validN = idx.Len() - trainN
	}
	if validLimit > 0 && validLimit < validN {
		validN = validLimit
	}
	if trainN <= 0 {
		return stats, fmt.Errorf("cnn-train: need at least one training image")
	}
	trainSamples := make([]data.FloatSample, 0, trainN)
	for i := 0; i < trainN; i++ {
		s, err := idx.Get(context.Background(), i)
		if err != nil {
			return stats, err
		}
		trainSamples = append(trainSamples, s)
	}
	validSamples := make([]data.FloatSample, 0, validN)
	for i := 0; i < validN; i++ {
		s, err := idx.Get(context.Background(), trainN+i)
		if err != nil {
			return stats, err
		}
		validSamples = append(validSamples, s)
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
	opt := newCNNTrainingOptimizer(m.module.NamedParameters(), "bias")
	defer opt.Close()
	baseDS := sliceDataset{samples: trainSamples}
	trainDS := data.TransformDataset[data.FloatSample]{Base: baseDS, Seed: 7, Apply: func(index int, seed uint64, sample data.FloatSample) (data.FloatSample, error) {
		if !augment {
			return sample, nil
		}
		out := data.FloatSample{Values: append([]float32(nil), sample.Values...), Label: sample.Label, Shape: append([]int(nil), sample.Shape...)}
		if err := data.RandomHorizontalFlipFloat32(out.Values, 1, h, w, seed, index, .5); err != nil {
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
			maxl := 0
			for _, l := range batch.Labels {
				if l > maxl {
					maxl = l
				}
			}

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

	// Final validation accuracy over validation set.
	if len(validSamples) > 0 {
		m.module.Train(false)
		x := make([]float32, len(validSamples)*h*w)
		labels := make([]int, len(validSamples))
		for i, s := range validSamples {
			copy(x[i*h*w:], s.Values)
			labels[i] = s.Label
		}
		xt, err := execution.New(x, []int{len(validSamples), 1, h, w}, device, false)
		if err != nil {
			return stats, err
		}
		defer xt.Close()
		var logits *autograd.Tensor
		execution.NoGrad(func() { logits = m.forward(xt) })
		defer logits.ReleaseGraph()
		values, err := logits.ToHost()
		if err != nil {
			return stats, err
		}
		correct := 0
		for i, want := range labels {
			got := 0
			best := values[i*nClasses]
			for c := 1; c < nClasses; c++ {
				if v := values[i*nClasses+c]; v > best {
					best = v
					got = c
				}
			}
			if got == want {
				correct++
			}
		}
		stats.accuracy = float32(correct) / float32(len(labels))
		var final *autograd.Tensor
		execution.NoGrad(func() { final = autograd.CrossEntropy(logits, labels, -1) })
		vv, err := final.ToHost()
		if err == nil {
			stats.finalLoss = vv[0]
		}
		final.ReleaseGraph()
	}
	if len(validSamples) == 0 {
		stats.finalLoss = stats.initialLoss
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
		restoredX, err := execution.New(make([]float32, 1*h*w), []int{1, 1, h, w}, device, false)
		if err == nil {
			restoredX.Close()
		}
	}
	fmt.Fprintf(log, "initial_loss=%.6f final_loss=%.6f accuracy=%.1f%% classes=%d samples=train%d/valid%d\n",
		stats.initialLoss, stats.finalLoss, 100*stats.accuracy, nClasses, trainN, validN)
	return stats, nil
}

func newCNNTrainingOptimizer(params []autograd.Parameter, suffix string) autograd.CheckpointOptimizer {
	decay := float32(.001)
	if suffix != "" {
		specs, err := autograd.PartitionParameters(params, func(name string) string {
			if strings.HasSuffix(name, suffix) {
				return "no_decay"
			}
			return "decay"
		})
		if err == nil {
			for i := range specs {
				if specs[i].Name == "no_decay" {
					specs[i].WeightDecay = 0
				} else {
					specs[i].WeightDecay = decay
				}
			}
			if opt, err := autograd.NewAdamWGroups(params, .01, specs); err == nil {
				return opt
			}
		}
	}
	return autograd.NewAdamW(params, .01, .001)
}
