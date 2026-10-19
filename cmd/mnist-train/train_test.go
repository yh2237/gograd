package main

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/data"
	"github.com/yh2237/gograd/tensor"
)

// Small local IDX data verifies scheduling/resume, not MNIST recognition quality.
func fixtureIDX(count int) ([]byte, []byte) {
	images := make([]byte, 16+count*784)
	labels := make([]byte, 8+count)
	for i, value := range []uint32{2051, uint32(count), 28, 28} {
		binary.BigEndian.PutUint32(images[i*4:], value)
	}
	binary.BigEndian.PutUint32(labels, 2049)
	binary.BigEndian.PutUint32(labels[4:], uint32(count))
	for n := 0; n < count; n++ {
		labels[8+n] = byte(n % 10)
		for p := 0; p < 784; p++ {
			images[16+n*784+p] = byte((p*7 + n*31) % 256)
		}
	}
	return images, labels
}
func writeDigits(t *testing.T, dir string) {
	t.Helper()
	trainImages, trainLabels := fixtureIDX(13)
	validImages, validLabels := fixtureIDX(7)
	for i, bytes := range [][]byte{trainImages, trainLabels, validImages, validLabels} {
		if err := os.WriteFile(filepath.Join(dir, mnistFiles[i]), bytes, 0600); err != nil {
			t.Fatal(err)
		}
	}
}
func TestMNISTMidEpochResume(t *testing.T) {
	for _, kind := range []string{"adamw", "sgd"} {
		for _, accum := range []int{1, 3} {
			for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
				t.Run(kind+"/"+string(device)+"/accum"+strconv.Itoa(accum), func(t *testing.T) {
					if device == tensor.CUDA && !cuda.Available() {
						t.Skip("CUDA unavailable")
					}
					dir := t.TempDir()
					writeDigits(t, dir)
					base := trainConfig{DataDir: dir, Device: string(device), Steps: 6, Batch: 4, Seed: 11, Optimizer: kind, Momentum: .9, Accum: accum}
					full := base
					full.Out = filepath.Join(dir, "full.safetensors")
					want, err := train(context.Background(), full, io.Discard)
					if err != nil {
						t.Fatal(err)
					}
					part := base
					part.Workers, part.Prefetch = 3, 3
					part.Out = filepath.Join(dir, "part.safetensors")
					part.StopAfter = 2
					if _, err := train(context.Background(), part, io.Discard); err != nil {
						t.Fatal(err)
					}
					resume := base
					resume.Workers, resume.Prefetch = 1, 2
					resume.Out = filepath.Join(dir, "resumed.safetensors")
					resume.Resume = part.Out + ".training.safetensors"
					got, err := train(context.Background(), resume, io.Discard)
					if err != nil {
						t.Fatal(err)
					}
					// The micro-batch count is checkpoint identity.
					changed := resume
					changed.Accum = accum + 1
					changed.Out = filepath.Join(dir, "changed.safetensors")
					if _, err := train(context.Background(), changed, io.Discard); err == nil {
						t.Fatal("changed accumulation accepted on resume")
					}
					if got.Steps != 6 || got.Accuracy != want.Accuracy || got.ReloadMaxAbs > 1e-5 {
						t.Fatalf("resume metrics %#v != %#v", got, want)
					}
					lossTolerance := float64(0)
					if device == tensor.CUDA {
						lossTolerance = 1e-6
					}
					if math.Abs(got.ValidationLoss-want.ValidationLoss) > lossTolerance {
						t.Fatal("resumed evaluation loss differed")
					}
					// On CUDA model/state readback must itself hold the OS thread.
					if device == tensor.CUDA {
						ctx, err := autograd.NewCUDAContext()
						if err != nil {
							t.Fatal(err)
						}
						defer ctx.Close()
					}
					a, err := newDigitModel(device, base.Seed)
					if err != nil {
						t.Fatal(err)
					}
					defer a.close()
					b, err := newDigitModel(device, base.Seed)
					if err != nil {
						t.Fatal(err)
					}
					defer b.close()
					for i, m := range []*digitModel{a, b} {
						file := full.Out
						if i == 1 {
							file = resume.Out
						}
						if err := m.module.LoadSafeTensors(file); err != nil {
							t.Fatal(err)
						}
					}
					left, right := a.module.StateDict(), b.module.StateDict()
					for name, values := range left {
						assertTrainingValues(t, device, name, values, right[name])
					}
					// Restore both combined checkpoints and compare optimizer buffers too.
					states := make([]autograd.OptimizerSnapshot, 2)
					schedules := make([]autograd.OneCycle, 2)
					for i, m := range []*digitModel{a, b} {
						opt, err := newTrainingOptimizer(m.module.NamedParameters(), base, .001)
						if err != nil {
							t.Fatal(err)
						}
						defer opt.Close()
						schedule := autograd.NewOneCycle(.005, 6, .3)
						file := full.Out
						if i == 1 {
							file = resume.Out
						}
						if _, err := autograd.LoadTrainingCheckpoint(file+".training.safetensors", &m.module, opt, schedule); err != nil {
							t.Fatal(err)
						}
						states[i], err = opt.Snapshot()
						if err != nil {
							t.Fatal(err)
						}
						schedules[i] = *schedule
					}
					if kind == "adamw" {
						for i, moments := range states[0].AdamW.Moments {
							assertTrainingValues(t, device, "AdamW M", moments.M, states[1].AdamW.Moments[i].M)
							assertTrainingValues(t, device, "AdamW V", moments.V, states[1].AdamW.Moments[i].V)
						}
						states[0].AdamW.Moments, states[1].AdamW.Moments = nil, nil
					} else {
						for i, buffer := range states[0].SGD.Buffers {
							assertTrainingValues(t, device, "SGD momentum", buffer, states[1].SGD.Buffers[i])
						}
						states[0].SGD.Buffers, states[1].SGD.Buffers = nil, nil
					}
					if !reflect.DeepEqual(states[0], states[1]) || !reflect.DeepEqual(schedules[0], schedules[1]) {
						t.Fatal("resume changed optimizer configuration or schedule")
					}
					metadata, _ := autograd.TrainingCheckpointMetadata(full.Out + ".training.safetensors")
					var finalState checkpointState
					if err := json.Unmarshal([]byte(metadata["data_state"]), &finalState); err != nil {
						t.Fatal(err)
					}
					other, _ := autograd.TrainingCheckpointMetadata(resume.Out + ".training.safetensors")
					if metadata["data_state"] != other["data_state"] {
						t.Fatal("resumed sampler state differed")
					}
					bad := resume
					bad.Batch++
					if _, err := train(context.Background(), bad, io.Discard); err == nil {
						t.Fatal("incompatible resume configuration accepted")
					}
					bad = resume
					bad.Checkpoint = bad.Out
					if _, err := train(context.Background(), bad, io.Discard); err == nil {
						t.Fatal("inference output could overwrite training state")
					}
				})
			}
		}
	}
}

func assertTrainingValues(t *testing.T, device tensor.Device, name string, left, right []float32) {
	t.Helper()
	if len(left) != len(right) {
		t.Fatalf("%s shape differed", name)
	}
	// Linear bias VJPs use atomicAdd on CUDA, so continuation is numerically
	// equivalent; CPU weights/moments and all sampler/scheduler state are exact.
	tolerance := float64(0)
	if device == tensor.CUDA {
		tolerance = 2e-7
	}
	maxAbs := float64(0)
	for i, v := range left {
		diff := math.Abs(float64(v) - float64(right[i]))
		maxAbs = math.Max(maxAbs, diff)
		if math.IsNaN(diff) || diff > tolerance {
			t.Fatalf("%s[%d] %g != %g (difference %g)", name, i, v, right[i], diff)
		}
	}
	t.Logf("%s resume max_abs=%g", name, maxAbs)
}

type cancelOnProgress struct{ cancel context.CancelFunc }

func (w cancelOnProgress) Write(p []byte) (int, error) {
	w.cancel()
	return len(p), nil
}

func TestMNISTCancelSavesCheckpoint(t *testing.T) {
	dir := t.TempDir()
	writeDigits(t, dir)
	base := trainConfig{DataDir: dir, Device: "cpu", Steps: 6, Batch: 4, Accum: 3, Seed: 11}
	full := base
	full.Out = filepath.Join(dir, "full.safetensors")
	if _, err := train(context.Background(), full, io.Discard); err != nil {
		t.Fatal(err)
	}
	part := base
	part.Out = filepath.Join(dir, "part.safetensors")
	part.Workers, part.Prefetch = 2, 3
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stats, err := train(ctx, part, cancelOnProgress{cancel})
	if !errors.Is(err, context.Canceled) || stats.Steps != 1 {
		t.Fatalf("stop stats=%+v err=%v", stats, err)
	}
	if _, err := os.Stat(part.Out); !os.IsNotExist(err) {
		t.Fatal("cancellation wrote inference weights")
	}
	resume := base
	resume.Out = part.Out
	resume.Resume = part.Out + ".training.safetensors"
	if _, err := train(context.Background(), resume, io.Discard); err != nil {
		t.Fatal(err)
	}
	assertCheckpointValues(t, tensor.CPU, full.Out+".training.safetensors", resume.Out+".training.safetensors")
}

// Cancellation after micro-batch 5 rewinds across an epoch boundary: with 13
// samples and batch size 4, group 2 includes the tail of epoch 0 and epoch 1.
func TestMNISTCanceledAccumulationResume(t *testing.T) {
	// The interruption point is the end of the first epoch, where the sampler
	// has to rewind across the epoch boundary.
	const after = 5
	for _, kind := range []string{"adamw", "sgd"} {
		for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
			for _, workers := range []int{0} {
				t.Run(kind+"/"+string(device)+"/workers"+strconv.Itoa(workers)+"/micro"+strconv.Itoa(after), func(t *testing.T) {
					if device == tensor.CUDA && !cuda.Available() {
						t.Skip("CUDA unavailable")
					}
					dir := t.TempDir()
					writeDigits(t, dir)
					base := trainConfig{DataDir: dir, Device: string(device), Steps: 6, Batch: 4, Accum: 3, Seed: 11, Optimizer: kind, Momentum: .9, MaxLR: .005}
					if kind == "adamw" {
						base.Momentum = 0
					}
					full := base
					full.Out = filepath.Join(dir, "full.safetensors")
					if _, err := train(context.Background(), full, io.Discard); err != nil {
						t.Fatal(err)
					}
					if device == tensor.CUDA {
						gpu, err := autograd.NewCUDAContext()
						if err != nil {
							t.Fatal(err)
						}
						defer gpu.Close()
					}
					model, err := newDigitModel(device, base.Seed)
					if err != nil {
						t.Fatal(err)
					}
					defer model.close()
					execution := autograd.NewExecutionContext()
					if err := execution.BindModule(&model.module); err != nil {
						t.Fatal(err)
					}
					dataset, _, err := readDigits(dir, mnistFiles[0], mnistFiles[1], 0)
					if err != nil {
						t.Fatal(err)
					}
					sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{Shuffle: true, Seed: base.Seed})
					if err != nil {
						t.Fatal(err)
					}
					loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](dataset, sampler, data.Stack, data.LoaderOptions{BatchSize: base.Batch, Workers: workers})
					if err != nil {
						t.Fatal(err)
					}
					defer loader.Close()
					schedule := autograd.NewOneCycle(base.MaxLR, base.Steps, .3)
					optimizer, err := newTrainingOptimizer(model.module.NamedParameters(), base, float32(schedule.LR()))
					if err != nil {
						t.Fatal(err)
					}
					defer optimizer.Close()
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					micro := 0
					var savepoint []byte
					step := trainingStep(model, execution, device)
					loop := autograd.TrainingLoop[data.FloatBatch]{
						Reader: loader, Sampler: sampler, Optimizer: optimizer, Schedule: schedule,
						Updates: base.Steps, Applied: schedule.StepCount, Accumulate: base.Accum,
						Step: func(ctx context.Context, batch data.FloatBatch, accumulator *autograd.GradAccumulator) (float64, bool, error) {
							loss, complete, err := step(ctx, batch, accumulator)
							if err != nil {
								return 0, false, err
							}
							micro++
							if micro == after {
								cancel()
							}
							return loss, complete, nil
						},
						Save: func(rewound []byte) error {
							savepoint = append([]byte(nil), rewound...)
							return nil
						},
					}
					if err := loop.Run(ctx); !errors.Is(err, context.Canceled) || savepoint == nil || schedule.StepCount != after/base.Accum {
						t.Fatalf("stop savepoint=%q steps=%d err=%v", savepoint, schedule.StepCount, err)
					}
					var rollback data.IndexSamplerState
					if err := json.Unmarshal(savepoint, &rollback); err != nil {
						t.Fatal(err)
					}
					if rollback.Epoch != 0 || rollback.Position != 12 || sampler.State().Epoch != 1 {
						t.Fatal("test did not rewind across epoch boundary")
					}
					state, err := loader.State()
					if err != nil {
						t.Fatal(err)
					}
					state.Sampler = savepoint
					metadata, err := autograd.TrainingCheckpointMetadata(full.Out + ".training.safetensors")
					if err != nil {
						t.Fatal(err)
					}
					var checkpoint checkpointState
					if err := json.Unmarshal([]byte(metadata["data_state"]), &checkpoint); err != nil {
						t.Fatal(err)
					}
					checkpoint.Loader = state
					payload, err := json.Marshal(checkpoint)
					if err != nil {
						t.Fatal(err)
					}
					metadata["data_state"] = string(payload)
					partPath := filepath.Join(dir, "part.training.safetensors")
					if err := autograd.SaveTrainingCheckpoint(partPath, &model.module, optimizer, schedule, metadata); err != nil {
						t.Fatal(err)
					}
					resume := base
					resume.Resume, resume.Out = partPath, filepath.Join(dir, "resumed.safetensors")
					resume.Workers = 1 // Worker settings need not match the interrupted run.
					if _, err := train(context.Background(), resume, io.Discard); err != nil {
						t.Fatal(err)
					}
					assertCheckpointValues(t, device, full.Out+".training.safetensors", resume.Out+".training.safetensors")
				})
			}
		}
	}
}

func assertCheckpointValues(t *testing.T, device tensor.Device, leftPath, rightPath string) {
	t.Helper()
	left, err := autograd.OpenSafeTensorFile(leftPath)
	if err != nil {
		t.Fatal(err)
	}
	defer left.Close()
	right, err := autograd.OpenSafeTensorFile(rightPath)
	if err != nil {
		t.Fatal(err)
	}
	defer right.Close()
	if !reflect.DeepEqual(left.Names(), right.Names()) || !reflect.DeepEqual(left.Metadata, right.Metadata) {
		t.Fatal("checkpoint metadata or tensor names changed after resume")
	}
	for _, name := range left.Names() {
		a, shapeA, err := left.ReadF32(name)
		if err != nil {
			t.Fatal(err)
		}
		b, shapeB, err := right.ReadF32(name)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(shapeA, shapeB) {
			t.Fatalf("%s shape changed", name)
		}
		assertTrainingValues(t, device, name, a, b)
	}
}

func TestMNISTLegacyAccumulationCheckpoint(t *testing.T) {
	dir := t.TempDir()
	writeDigits(t, dir)
	base := trainConfig{DataDir: dir, Device: "cpu", Steps: 6, Batch: 4, Seed: 11, MaxLR: .005, Optimizer: "adamw"}
	full := base
	full.Out = filepath.Join(dir, "full.safetensors")
	if _, err := train(context.Background(), full, io.Discard); err != nil {
		t.Fatal(err)
	}
	part := base
	part.StopAfter, part.Out = 2, filepath.Join(dir, "part.safetensors")
	if _, err := train(context.Background(), part, io.Discard); err != nil {
		t.Fatal(err)
	}
	path := part.Out + ".training.safetensors"
	model, err := newDigitModel(tensor.CPU, base.Seed)
	if err != nil {
		t.Fatal(err)
	}
	defer model.close()
	optimizer, err := newTrainingOptimizer(model.module.NamedParameters(), base, .001)
	if err != nil {
		t.Fatal(err)
	}
	defer optimizer.Close()
	schedule := autograd.NewOneCycle(base.MaxLR, base.Steps, .3)
	metadata, err := autograd.LoadTrainingCheckpoint(path, &model.module, optimizer, schedule)
	if err != nil {
		t.Fatal(err)
	}
	var state map[string]json.RawMessage
	if err := json.Unmarshal([]byte(metadata["data_state"]), &state); err != nil {
		t.Fatal(err)
	}
	var identity map[string]json.RawMessage
	if err := json.Unmarshal(state["Identity"], &identity); err != nil {
		t.Fatal(err)
	}
	delete(identity, "Accum")
	state["Identity"], err = json.Marshal(identity)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	metadata["data_state"] = string(payload)
	if err := autograd.SaveTrainingCheckpoint(path, &model.module, optimizer, schedule, metadata); err != nil {
		t.Fatal(err)
	}
	resume := base
	resume.Resume, resume.Out = path, filepath.Join(dir, "resumed.safetensors")
	if _, err := train(context.Background(), resume, io.Discard); err != nil {
		t.Fatal(err)
	}
	assertCheckpointValues(t, tensor.CPU, full.Out+".training.safetensors", resume.Out+".training.safetensors")
}

func TestMNISTDownload(t *testing.T) {
	images, labels := fixtureIDX(2)
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		z := gzip.NewWriter(w)
		defer z.Close()
		if filepath.Base(r.URL.Path) == mnistFiles[1]+".gz" || filepath.Base(r.URL.Path) == mnistFiles[3]+".gz" {
			z.Write(labels)
		} else {
			z.Write(images)
		}
	}))
	defer server.Close()
	dir := t.TempDir()
	if err := downloadMNIST(context.Background(), dir, server.URL); err != nil {
		t.Fatal(err)
	}
	if _, _, err := readDigits(dir, mnistFiles[0], mnistFiles[1], 0); err != nil {
		t.Fatal(err)
	}
	if err := downloadMNIST(context.Background(), dir, server.URL); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 4 {
		t.Fatal("existing IDX files were overwritten/redownloaded")
	}
	failed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(500) }))
	defer failed.Close()
	path := filepath.Join(dir, "failed.gz")
	if err := downloadFile(context.Background(), http.DefaultClient, failed.URL, path); err == nil {
		t.Fatal("failed download accepted")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("failed download left a final file")
	}
}
