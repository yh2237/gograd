package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
)

func featureFixture(t *testing.T) string {
	t.Helper()
	var raw bytes.Buffer
	header := map[string]any{}
	names := []string{"one", "two", "three", "four"}
	nameJSON, _ := json.Marshal(names)
	header["__metadata__"] = map[string]string{"format": "gograd-speech-timing-features-1", "ids": string(nameJSON)}
	for i := range names {
		frames := 7 + i
		prefix := strings.Repeat("0", 4) + string(rune('0'+i)) + "."
		start := raw.Len()
		for f := 0; f < frames; f++ {
			for j := 0; j < 3; j++ {
				binary.Write(&raw, binary.LittleEndian, int64(3+(i+f+j)%20))
			}
		}
		header[prefix+"ids"] = tensorEntry{"I64", []int{frames, 3}, [2]int{start, raw.Len()}}
		start = raw.Len()
		for f := 0; f < frames; f++ {
			for j := 0; j < 15; j++ {
				binary.Write(&raw, binary.LittleEndian, float32((f+j)%7)/7)
			}
		}
		header[prefix+"cont"] = tensorEntry{"F32", []int{frames, 15}, [2]int{start, raw.Len()}}
		start = raw.Len()
		for f := 0; f < frames; f++ {
			for j := 0; j < 80; j++ {
				binary.Write(&raw, binary.LittleEndian, float32(math.Sin(float64(i+f+j)*.1)))
			}
		}
		header[prefix+"target"] = tensorEntry{"F32", []int{frames, 80}, [2]int{start, raw.Len()}}
	}
	data, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var file bytes.Buffer
	binary.Write(&file, binary.LittleEndian, uint64(len(data)))
	file.Write(data)
	file.Write(raw.Bytes())
	path := filepath.Join(t.TempDir(), "features.safetensors")
	if err := os.WriteFile(path, file.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestTrainerResumeMatchesUninterrupted(t *testing.T) {
	cache := featureFixture(t)
	for _, device := range []string{"cpu", "cuda"} {
		for _, context := range []bool{false, true} {
			t.Run(device+"/"+map[bool]string{false: "base", true: "context"}[context], func(t *testing.T) {
				if device == "cuda" && !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				c := trainConfig{Cache: cache, Steps: 6, Valid: 1, Batch: 1, Window: 5, EvalEvery: 2, CheckpointEvery: 99,
					Seed: 17, Context: context, Device: device, ModelID: "test"}
				c.Out = filepath.Join(t.TempDir(), "full.safetensors")
				if err := train(c); err != nil {
					t.Fatal(err)
				}
				full := c.Out
				if device == "cuda" {
					c.Out = filepath.Join(t.TempDir(), "second-full.safetensors")
					if err := train(c); err != nil {
						t.Fatal(err)
					}
					want, _ := os.ReadFile(full + ".training.safetensors")
					got, _ := os.ReadFile(c.Out + ".training.safetensors")
					baselineError := compareCUDACheckpoint(t, want, got)
					t.Logf("CUDA uninterrupted rerun max absolute error = %g", baselineError)
					if baselineError > 1e-6 {
						t.Fatalf("CUDA baseline error %g exceeds 1e-6", baselineError)
					}
				}
				c.Out = filepath.Join(t.TempDir(), "partial.safetensors")
				c.StopAfter = 3
				if err := train(c); err != nil {
					t.Fatal(err)
				}
				partial := c.Out
				c.Resume = partial + ".training.safetensors"
				c.Out = filepath.Join(t.TempDir(), "resumed.safetensors")
				c.StopAfter = 0
				if err := train(c); err != nil {
					t.Fatal(err)
				}
				for _, suffix := range []string{"", ".training.safetensors"} {
					want, err := os.ReadFile(full + suffix)
					if err != nil {
						t.Fatal(err)
					}
					got, err := os.ReadFile(c.Out + suffix)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(want, got) {
						if device == "cpu" {
							t.Fatalf("resumed checkpoint differs byte-for-byte: %s", suffix)
						}
						maxError := compareCUDACheckpoint(t, want, got)
						t.Logf("CUDA checkpoint %s max absolute error = %g", suffix, maxError)
						if maxError > 1e-6 {
							t.Fatalf("CUDA resume error %g exceeds 1e-6", maxError)
						}
					}
				}
				metadata, err := autograd.TrainingCheckpointMetadata(c.Out + ".training.safetensors")
				if err != nil {
					t.Fatal(err)
				}
				var state trainerState
				if err := json.Unmarshal([]byte(metadata["trainer"]), &state); err != nil {
					t.Fatal(err)
				}
				if state.BestStep < 0 || state.BestStep >= 6 || len(state.Order) != 4 {
					t.Fatal("missing trainer state")
				}
				// Changed schedule/cache must be rejected without creating inference output.
				c.Out = filepath.Join(t.TempDir(), "bad.safetensors")
				c.Steps++
				if err := train(c); err == nil {
					t.Fatal("accepted changed total steps")
				}
				if _, err := os.Stat(c.Out); !os.IsNotExist(err) {
					t.Fatal("invalid resume wrote output")
				}
				c.Steps--
				c.Cache = featureFixture(t)
				file, err := os.OpenFile(c.Cache, os.O_APPEND|os.O_WRONLY, 0600)
				if err != nil {
					t.Fatal(err)
				}
				file.Write([]byte{0})
				file.Close()
				if err := train(c); err == nil {
					t.Fatal("accepted changed feature cache")
				}
			})
		}
	}
}

// CUDA's atomic gradient reductions can reorder additions between executions.
// Check every tensor with a float32 tolerance; sampler/scheduler/split remain exact.
func compareCUDACheckpoint(t *testing.T, want, got []byte) float64 {
	t.Helper()
	decode := func(file []byte) (map[string]tensorEntry, map[string]string, []byte) {
		n := int(binary.LittleEndian.Uint64(file[:8]))
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(file[8:8+n], &raw); err != nil {
			t.Fatal(err)
		}
		var metadata map[string]string
		if err := json.Unmarshal(raw["__metadata__"], &metadata); err != nil {
			t.Fatal(err)
		}
		delete(raw, "__metadata__")
		entries := map[string]tensorEntry{}
		for name, entry := range raw {
			var value tensorEntry
			if err := json.Unmarshal(entry, &value); err != nil {
				t.Fatal(err)
			}
			entries[name] = value
		}
		return entries, metadata, file[8+n:]
	}
	a, am, ad := decode(want)
	b, bm, bd := decode(got)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("CUDA checkpoint tensor layout differs")
	}
	if am["format"] == "gograd-training-1" {
		var au, bu map[string]string
		json.Unmarshal([]byte(am["user"]), &au)
		json.Unmarshal([]byte(bm["user"]), &bu)
		var as, bs trainerState
		json.Unmarshal([]byte(au["trainer"]), &as)
		json.Unmarshal([]byte(bu["trainer"]), &bs)
		if math.Abs(as.Best-bs.Best) > 1e-6 {
			t.Fatal("CUDA best validation score diverged")
		}
		bs.Best = as.Best
		if !reflect.DeepEqual(as, bs) {
			t.Fatal("CUDA trainer/sampler state diverged")
		}
		delete(am, "user")
		delete(bm, "user")
	}
	if !reflect.DeepEqual(am, bm) {
		t.Fatal("CUDA checkpoint metadata diverged")
	}
	maximum := 0.0
	for _, entry := range a {
		for i := entry.Offsets[0]; i < entry.Offsets[1]; i += 4 {
			x := math.Float32frombits(binary.LittleEndian.Uint32(ad[i:]))
			y := math.Float32frombits(binary.LittleEndian.Uint32(bd[i:]))
			if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) || math.IsInf(float64(x), 0) || math.IsInf(float64(y), 0) {
				t.Fatal("non-finite CUDA checkpoint tensor")
			}
			maximum = math.Max(maximum, math.Abs(float64(x-y)))
		}
	}
	return maximum
}

func TestSampleBatchPadsAndDropsContext(t *testing.T) {
	item := utterance{Frames: 1, Continuous: 15, IDs: []int{3, 4, 5}, Cont: make([]float32, 15), Target: make([]float32, 80)}
	for i := range item.Cont {
		item.Cont[i] = 1
	}
	sampler, err := autograd.NewWindowSampler([]int{1}, 4, 1, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	ids, cont, target := sampleBatch([]utterance{item}, sampler, 15, 1, 4)
	if ids[0] != 3 || cont[0] != 1 || cont[4] != 0 || cont[12] != 0 || cont[13] != 1 || cont[14] != 1 {
		t.Fatal("sampled frame/context mask changed")
	}
	for _, value := range target[80:] {
		if !math.IsNaN(float64(value)) {
			t.Fatal("padding was not masked")
		}
	}
}
