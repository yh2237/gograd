package main

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
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
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA && !cuda.Available() {
				t.Skip("CUDA unavailable")
			}
			dir := t.TempDir()
			writeDigits(t, dir)
			base := trainConfig{DataDir: dir, Device: string(device), Steps: 6, Batch: 4, Seed: 11}
			full := base
			full.Out = filepath.Join(dir, "full.safetensors")
			want, err := train(context.Background(), full, io.Discard)
			if err != nil {
				t.Fatal(err)
			}
			part := base
			part.Out = filepath.Join(dir, "part.safetensors")
			part.StopAfter = 2
			if _, err := train(context.Background(), part, io.Discard); err != nil {
				t.Fatal(err)
			}
			resume := base
			resume.Out = filepath.Join(dir, "resumed.safetensors")
			resume.Resume = part.Out + ".training.safetensors"
			got, err := train(context.Background(), resume, io.Discard)
			if err != nil {
				t.Fatal(err)
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
			// Restore both combined checkpoints and compare AdamW moments too.
			states := make([]autograd.OptimizerState, 2)
			schedules := make([]autograd.OneCycle, 2)
			for i, m := range []*digitModel{a, b} {
				opt := autograd.NewAdamW(m.module.NamedParameters(), .001, 0)
				defer opt.Close()
				schedule := autograd.NewOneCycle(.005, 6, .3)
				file := full.Out
				if i == 1 {
					file = resume.Out
				}
				if _, err := autograd.LoadTrainingCheckpoint(file+".training.safetensors", &m.module, opt, schedule); err != nil {
					t.Fatal(err)
				}
				states[i], err = opt.State()
				if err != nil {
					t.Fatal(err)
				}
				schedules[i] = *schedule
			}
			for i, moments := range states[0].Moments {
				assertTrainingValues(t, device, "AdamW M", moments.M, states[1].Moments[i].M)
				assertTrainingValues(t, device, "AdamW V", moments.V, states[1].Moments[i].V)
			}
			states[0].Moments, states[1].Moments = nil, nil
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
