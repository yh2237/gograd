package autograd

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func testParameter(t *testing.T, values []float32) Parameter {
	t.Helper()
	value, err := Zeros([]int{len(values)}, tensor.CPU, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := value.CopyFrom(values); err != nil {
		t.Fatal(err)
	}
	return Parameter{Name: "weight", Value: value}
}

func drive(t *testing.T, optimizer *AdamW, parameter Parameter, grads [][]float32) {
	t.Helper()
	for _, grad := range grads {
		parameter.Value.Grad = append([]float32(nil), grad...)
		optimizer.Step()
	}
}

func TestAdamWStateResumesTraining(t *testing.T) {
	first := testParameter(t, []float32{0.5, -0.25, 1.5})
	optimizer := NewAdamW([]Parameter{first}, 0.1, 0.01)
	drive(t, optimizer, first, [][]float32{{0.3, -0.1, 0.2}, {-0.2, 0.4, -0.3}, {0.1, 0.1, -0.1}})

	state, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	if state.StepCount != 3 {
		t.Fatalf("step count = %d, want 3", state.StepCount)
	}

	second := testParameter(t, append([]float32(nil), first.Value.Data...))
	resumed := NewAdamW([]Parameter{second}, 0.1, 0.01)
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	nextGrads := [][]float32{{0.2, -0.2, 0.3}, {-0.1, 0.2, 0.1}}
	drive(t, optimizer, first, nextGrads)
	drive(t, resumed, second, nextGrads)
	if !reflect.DeepEqual(first.Value.Data, second.Value.Data) {
		t.Fatalf("resumed weights = %v, want %v", second.Value.Data, first.Value.Data)
	}
}

func TestAdamWStateSafeTensorsRoundTrip(t *testing.T) {
	first := testParameter(t, []float32{0.5, -0.25})
	optimizer := NewAdamW([]Parameter{first}, 0.1, 0.01)
	drive(t, optimizer, first, [][]float32{{0.3, -0.1}, {-0.2, 0.4}})

	path := filepath.Join(t.TempDir(), "optimizer.safetensors")
	if err := optimizer.SaveSafeTensors(path); err != nil {
		t.Fatal(err)
	}
	second := testParameter(t, []float32{0, 0})
	restored := NewAdamW([]Parameter{second}, 0.1, 0.01)
	if err := restored.LoadSafeTensors(path); err != nil {
		t.Fatal(err)
	}
	want, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	got, err := restored.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("restored state differs: %+v vs %+v", got, want)
	}
}

func TestAdamWStateCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	value, err := Zeros([]int{3}, tensor.CUDA, true)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	parameter := Parameter{Name: "weight", Value: value}
	optimizer := NewAdamW([]Parameter{parameter}, 0.1, 0.01)
	defer optimizer.Close()
	if err := parameter.Value.ensureGradGPU().CopyFromHost(floatBytes([]float32{0.3, -0.1, 0.2})); err != nil {
		t.Fatal(err)
	}
	optimizer.Step()
	state, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	if state.StepCount != 1 || len(state.Moments) != 1 || len(state.Moments[0].M) != 3 {
		t.Fatalf("unexpected CUDA state: %+v", state)
	}
	target := testParameter(t, []float32{0, 0, 0})
	restored := NewAdamW([]Parameter{target}, 0.1, 0.01)
	if err := restored.LoadState(state); err != nil {
		t.Fatal(err)
	}
}

func TestSetTrainingPropagatesDropout(t *testing.T) {
	layer := &DropoutLayer{Probability: 0.5, Training: true}
	SetTraining(false, layer, nil)
	if layer.Training {
		t.Fatal("dropout was not disabled")
	}
	SetTraining(true, layer)
	if !layer.Training {
		t.Fatal("dropout was not enabled")
	}
}

func TestIndexLoaderIsDeterministic(t *testing.T) {
	first := NewIndexLoader(10, 4, 42)
	second := NewIndexLoader(10, 4, 42)
	firstEpoch := first.Epoch()
	if !reflect.DeepEqual(firstEpoch, second.Epoch()) {
		t.Fatal("same seed produced different epochs")
	}
	if reflect.DeepEqual(firstEpoch, first.Epoch()) {
		t.Fatal("next epoch reused the same permutation")
	}
	seen := map[int]bool{}
	for _, batch := range firstEpoch {
		if len(batch) == 0 || len(batch) > 4 {
			t.Fatalf("invalid batch size %d", len(batch))
		}
		for _, index := range batch {
			if index < 0 || index >= 10 || seen[index] {
				t.Fatalf("invalid or repeated index %d", index)
			}
			seen[index] = true
		}
	}
	if len(seen) != 10 {
		t.Fatalf("epoch covered %d indices, want 10", len(seen))
	}
}
