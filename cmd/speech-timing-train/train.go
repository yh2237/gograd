package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type trainerState struct {
	Version                                            int
	CacheSHA256                                        string
	Steps, Valid, Continuous, Batch, Window, EvalEvery int
	Seed                                               int64
	ModelID                                            string
	Order                                              []int
	Sampler                                            autograd.WindowSamplerState
	Best                                               float64
	BestStep                                           int
}

func train(c trainConfig) error {
	return trainContext(context.Background(), c)
}

func trainContext(ctx context.Context, c trainConfig) error {
	if c.Steps < 2 || c.Batch < 1 || c.Window < 1 || c.EvalEvery < 1 || c.CheckpointEvery < 1 || c.StopAfter < 0 || c.StopAfter > c.Steps {
		return fmt.Errorf("invalid training steps, window, batch or interval")
	}
	if c.Checkpoint == "" {
		c.Checkpoint = c.Out + ".training.safetensors"
	}
	outAbs, err := filepath.Abs(c.Out)
	if err != nil {
		return err
	}
	cacheAbs, err := filepath.Abs(c.Cache)
	if err != nil {
		return err
	}
	checkpointAbs, err := filepath.Abs(c.Checkpoint)
	if err != nil {
		return err
	}
	resumeAbs := ""
	if c.Resume != "" {
		resumeAbs, err = filepath.Abs(c.Resume)
		if err != nil {
			return err
		}
	}
	if outAbs == cacheAbs || outAbs == checkpointAbs || outAbs == resumeAbs || cacheAbs == checkpointAbs || cacheAbs == resumeAbs {
		return fmt.Errorf("inference output, feature cache and training checkpoint must use different paths")
	}
	if err := requireNewPath(c.Out); err != nil {
		return err
	}
	if c.Resume == "" || checkpointAbs != resumeAbs {
		if err := requireNewPath(c.Checkpoint); err != nil {
			return err
		}
	}
	hash, err := cacheDigest(c.Cache)
	if err != nil {
		return err
	}
	items, err := readFeatures(c.Cache)
	if err != nil {
		return fmt.Errorf("read features: %w", err)
	}
	if c.Valid < 1 || c.Valid >= len(items) {
		return fmt.Errorf("invalid validation count")
	}
	continuous := 4
	if c.Context {
		continuous = 15
	}
	for _, item := range items {
		if item.Frames < 1 || item.Continuous < continuous {
			return fmt.Errorf("cache %s has incompatible frames/features", item.ID)
		}
	}
	state := trainerState{Version: 1, CacheSHA256: hash, Steps: c.Steps, Valid: c.Valid, Continuous: continuous,
		Batch: c.Batch, Window: c.Window, EvalEvery: c.EvalEvery, Seed: c.Seed, ModelID: c.ModelID, Best: math.Inf(1), BestStep: -1}
	if c.Resume != "" {
		metadata, err := autograd.TrainingCheckpointMetadata(c.Resume)
		if err != nil {
			return err
		}
		var saved trainerState
		if err := json.Unmarshal([]byte(metadata["trainer"]), &saved); err != nil {
			return fmt.Errorf("trainer state: %w", err)
		}
		if saved.Version != state.Version || saved.CacheSHA256 != hash || saved.Steps != c.Steps || saved.Valid != c.Valid || saved.Continuous != continuous || saved.Batch != c.Batch || saved.Window != c.Window || saved.EvalEvery != c.EvalEvery || saved.Seed != c.Seed || saved.ModelID != c.ModelID || saved.BestStep < 0 || saved.BestStep >= c.Steps || saved.Best < 0 || math.IsNaN(saved.Best) || math.IsInf(saved.Best, 0) {
			return fmt.Errorf("resume checkpoint does not match cache/configuration; keep --steps, seed, split, context and batch/window unchanged")
		}
		state = saved
	} else {
		order, err := readFeatureSplit(c.Cache, c.Seed, len(items))
		if err != nil {
			return err
		}
		if order == nil {
			order = rand.New(rand.NewSource(c.Seed)).Perm(len(items))
			fmt.Println("split=Go RNG (cache has no matching Python split)")
		} else {
			fmt.Println("split=Python RNG from feature cache")
		}
		state.Order = order
	}
	if err := validateOrder(state.Order, len(items)); err != nil {
		return err
	}
	validation, training := make([]utterance, c.Valid), make([]utterance, len(items)-c.Valid)
	for i, index := range state.Order {
		if i < c.Valid {
			validation[i] = items[index]
		} else {
			training[i-c.Valid] = items[index]
		}
	}
	frames := make([]int, len(training))
	for i, item := range training {
		frames[i] = item.Frames
	}
	dropProbability := 0.0
	if c.Context {
		dropProbability = .15
	}
	sampler, err := autograd.NewWindowSampler(frames, c.Window, c.Batch, dropProbability, uint64(c.Seed))
	if err != nil {
		return err
	}
	if c.Resume != "" {
		if err := sampler.LoadState(state.Sampler); err != nil {
			return err
		}
	}
	device := tensor.CPU
	switch c.Device {
	case "cpu":
	case "auto":
		if cuda.Available() {
			device = tensor.CUDA
		}
	case "cuda":
		if !cuda.Available() {
			return fmt.Errorf("CUDA unavailable")
		}
		device = tensor.CUDA
	default:
		return fmt.Errorf("unknown device %q", c.Device)
	}
	if device == tensor.CUDA {
		ctx, err := autograd.NewCUDAContext()
		if err != nil {
			return err
		}
		defer ctx.Close()
	}
	model, err := autograd.NewSpeechTiming(continuous, device, c.Seed)
	if err != nil {
		return err
	}
	defer closeModule(&model.Module)
	bestModel, err := autograd.NewSpeechTiming(continuous, tensor.CPU, c.Seed)
	if err != nil {
		return err
	}
	defer closeModule(&bestModel.Module)
	combined := &autograd.Module{Children: []autograd.NamedModule{{Name: "current", Module: &model.Module}, {Name: "best", Module: &bestModel.Module}}}
	params := model.Parameters()
	opt := autograd.NewAdamW(params, .002, .0001)
	defer opt.Close()
	schedule := autograd.NewOneCycle(.002, c.Steps, .1)
	opt.LR = float32(schedule.LR())
	if c.Resume != "" {
		if _, err := autograd.LoadTrainingCheckpoint(c.Resume, combined, opt, schedule); err != nil {
			return err
		}
		if schedule.Total != c.Steps || state.BestStep >= opt.StepCount {
			return fmt.Errorf("inconsistent resume step")
		}
		if err := os.MkdirAll(filepath.Dir(c.Out), 0700); err != nil {
			return err
		}
		if err := bestModel.Module.SaveSafeTensorsMetadata(c.Out, checkpointMetadata(c.ModelID, c.Context, state.Best, state.BestStep)); err != nil {
			return err
		}
		fmt.Printf("resumed step=%d best=%.6f@%d\n", opt.StepCount, state.Best, state.BestStep)
	}
	model.Train(true)
	limit := c.Steps
	if c.StopAfter > 0 {
		limit = c.StopAfter
	}
	if limit < opt.StepCount {
		return fmt.Errorf("--stop-after is before the saved step")
	}
	started := time.Now()
	save := func() error {
		state.Sampler = sampler.State()
		data, err := json.Marshal(state)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(c.Checkpoint), 0700); err != nil {
			return err
		}
		return autograd.SaveTrainingCheckpoint(c.Checkpoint, combined, opt, schedule, map[string]string{"trainer": string(data)})
	}
	fmt.Printf("device=%s utterances=%d train=%d valid=%d frames=%d steps=%d batch=%d window=%d output=%s\n", device, len(items), len(training), len(validation), totalFrames(items), c.Steps, c.Batch, c.Window, c.Out)
	for step := opt.StepCount; step < limit; step++ {
		if ctx.Err() != nil {
			if opt.StepCount == 0 {
				return ctx.Err()
			}
			fmt.Println("interrupted: saving at completed update", opt.StepCount)
			break
		}
		ids, cv, tv := sampleBatch(training, sampler, continuous, c.Batch, c.Window)
		lossValue, err := update(model, opt, schedule, ids, cv, tv, c.Batch, c.Window, continuous, device, uint32(c.Seed)+uint32(step)*8)
		if err != nil {
			return err
		}
		if step%c.EvalEvery == 0 || step == c.Steps-1 {
			score, err := evaluate(model, validation, continuous, device)
			if err != nil {
				return err
			}
			if math.IsNaN(score) || math.IsInf(score, 0) {
				return fmt.Errorf("non-finite validation score")
			}
			if score < state.Best {
				state.Best, state.BestStep = score, step
				if err := bestModel.Module.LoadStateDict(model.Module.StateDict()); err != nil {
					return err
				}
				if err := os.MkdirAll(filepath.Dir(c.Out), 0700); err != nil {
					return err
				}
				if err := bestModel.Module.SaveSafeTensorsMetadata(c.Out, checkpointMetadata(c.ModelID, c.Context, state.Best, state.BestStep)); err != nil {
					return err
				}
			}
			fmt.Printf("step=%d loss=%.5f valid=%.5f best=%.5f@%d elapsed=%s\n", step, lossValue, score, state.Best, state.BestStep, time.Since(started).Round(time.Second))
		}
		if opt.StepCount%c.CheckpointEvery == 0 && opt.StepCount != limit {
			if err := save(); err != nil {
				return err
			}
		}
	}
	if err := save(); err != nil {
		return err
	}
	fmt.Printf("finished completed=%d planned=%d best_valid_l1=%.6f best_step=%d wall=%s checkpoint=%s\n", opt.StepCount, c.Steps, state.Best, state.BestStep, time.Since(started).Round(time.Millisecond), c.Checkpoint)
	return nil
}

func update(model *autograd.SpeechTiming, opt *autograd.AdamW, schedule *autograd.OneCycle, ids []int, cv, tv []float32, batch, window, continuous int, device tensor.Device, dropoutSeed uint32) (float32, error) {
	cont, err := autograd.New(cv, []int{batch, window, continuous}, device, false)
	if err != nil {
		return 0, err
	}
	defer cont.Close()
	target, err := autograd.New(tv, []int{batch, window, 80}, device, false)
	if err != nil {
		return 0, err
	}
	defer target.Close()
	opt.ZeroGrad()
	loss := autograd.MaskedLoss(model.Forward(ids, cont, dropoutSeed), target, false)
	defer loss.ReleaseGraph()
	values, err := loss.ToHost()
	if err != nil {
		return 0, err
	}
	if math.IsNaN(float64(values[0])) || math.IsInf(float64(values[0]), 0) {
		return 0, fmt.Errorf("non-finite training loss")
	}
	if err := loss.Backward(); err != nil {
		return 0, err
	}
	autograd.ClipGradNorm(opt.Params, 1)
	opt.Step()
	schedule.Step(opt)
	return values[0], nil
}

func sampleBatch(items []utterance, sampler *autograd.WindowSampler, c, batch, window int) ([]int, []float32, []float32) {
	ids := make([]int, batch*window*3)
	cont := make([]float32, batch*window*c)
	target := make([]float32, batch*window*80)
	for b, sampled := range sampler.Batch() {
		item := items[sampled.Index]
		for t := 0; t < window; t++ {
			dst, src := b*window+t, sampled.Start+t
			if src >= item.Frames {
				for j := 0; j < 80; j++ {
					target[dst*80+j] = float32(math.NaN())
				}
				continue
			}
			copy(ids[dst*3:dst*3+3], item.IDs[src*3:src*3+3])
			copy(cont[dst*c:dst*c+c], item.Cont[src*item.Continuous:src*item.Continuous+c])
			if sampled.DropContext {
				for j := 4; j < min(13, c); j++ {
					cont[dst*c+j] = 0
				}
			}
			copy(target[dst*80:dst*80+80], item.Target[src*80:src*80+80])
		}
	}
	return ids, cont, target
}

func evaluate(model *autograd.SpeechTiming, items []utterance, c int, device tensor.Device) (float64, error) {
	wasTraining := model.Module.Training
	model.Train(false)
	defer model.Train(wasTraining)
	var scores float64
	for _, item := range items {
		cv := make([]float32, item.Frames*c)
		for t := 0; t < item.Frames; t++ {
			copy(cv[t*c:(t+1)*c], item.Cont[t*item.Continuous:t*item.Continuous+c])
		}
		cont, err := autograd.New(cv, []int{1, item.Frames, c}, device, false)
		if err != nil {
			return 0, err
		}
		var pred *autograd.Tensor
		autograd.NoGrad(func() { pred = model.Forward(item.IDs, cont, 0) })
		values, err := pred.ToHost()
		pred.ReleaseGraph()
		cont.Close()
		if err != nil {
			return 0, err
		}
		var total float64
		count := 0
		for t := 0; t < item.Frames; t++ {
			if item.IDs[t*3] == 2 {
				continue
			}
			for j := 0; j < 80; j++ {
				total += math.Abs(float64(values[t*80+j] - item.Target[t*80+j]))
				count++
			}
		}
		if count > 0 {
			scores += total / float64(count)
		}
	}
	return scores / float64(len(items)), nil
}

func requireNewPath(path string) error {
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("refusing to overwrite %s", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func cacheDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func validateOrder(order []int, count int) error {
	if len(order) != count {
		return fmt.Errorf("resume split length mismatch")
	}
	sorted := slices.Clone(order)
	slices.Sort(sorted)
	for i, n := range sorted {
		if i != n {
			return fmt.Errorf("resume split is not a permutation")
		}
	}
	return nil
}

func closeModule(module *autograd.Module) {
	for _, p := range module.NamedParameters() {
		p.Value.Close()
	}
}
