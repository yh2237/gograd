package autograd

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

type lossFixture struct {
	Kind       string      `json:"kind"`
	Reduction  string      `json:"reduction"`
	Weight     bool        `json:"weight"`
	Prediction [][]float32 `json:"prediction"`
	Target     [][]float32 `json:"target"`
	Weights    [][]float32 `json:"weights"`
	Loss       float32     `json:"loss"`
	Grad       [][]float32 `json:"grad"`
}

func lossFixtures(t *testing.T) []lossFixture {
	t.Helper()
	raw, err := os.ReadFile("../testdata/losses.json")
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Cases []lossFixture `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("empty loss fixtures")
	}
	return file.Cases
}

func flattened(values [][]float32) []float32 {
	out := make([]float32, 0, len(values)*len(values[0]))
	for _, row := range values {
		out = append(out, row...)
	}
	return out
}

func TestLossPyTorchParity(t *testing.T) {
	for _, fixture := range lossFixtures(t) {
		name := fixture.Kind + "/" + fixture.Reduction
		if fixture.Weight {
			name += "/weighted"
		}
		t.Run(name, func(t *testing.T) {
			pool2dDevices(t, func(t *testing.T, device tensor.Device) {
				pool2dContext(t, device)
				shape := []int{len(fixture.Prediction), len(fixture.Prediction[0])}
				prediction := conv2dTestTensor(t, flattened(fixture.Prediction), shape, device, true)
				target := conv2dTestTensor(t, flattened(fixture.Target), shape, device, false)
				options := LossOptions{}
				if fixture.Reduction == "sum" {
					options.Reduction = ReductionSum
				}
				if fixture.Weight {
					options.Weight = conv2dTestTensor(t, flattened(fixture.Weights), shape, device, false)
				}
				var loss *Tensor
				switch fixture.Kind {
				case "mse":
					loss = MSELoss(prediction, target, options)
				case "l1":
					loss = L1Loss(prediction, target, options)
				case "bce_logits":
					loss = BCEWithLogitsLoss(prediction, target, options)
				default:
					t.Fatalf("unknown loss kind %q", fixture.Kind)
				}
				defer loss.ReleaseGraph()
				value, err := loss.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "loss", value, []float32{fixture.Loss}, 5e-6, 3e-6)
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "gradient", conv2dHost(t, prediction, true), flattened(fixture.Grad), 5e-6, 3e-6)
			})
		})
	}
}

func TestLossReductionAndMasking(t *testing.T) {
	prediction := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{4}, tensor.CPU, true)
	target := conv2dTestTensor(t, []float32{0, 2, 0, 4}, []int{4}, tensor.CPU, false)
	// squared differences are [1, 0, 9, 0]; a zero weight drops element 2 from
	// both the sum and the mean's divisor.
	weights := conv2dTestTensor(t, []float32{1, 1, 0, 1}, []int{4}, tensor.CPU, false)
	sum := MSELoss(prediction, target, LossOptions{Reduction: ReductionSum, Weight: weights})
	defer sum.ReleaseGraph()
	value, err := sum.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "weighted sum", value, []float32{1}, 0, 0)
	mean := MSELoss(prediction, target, LossOptions{Weight: weights})
	defer mean.ReleaseGraph()
	value, err = mean.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "weighted mean", value, []float32{1.0 / 3}, 1e-7, 1e-6)
	// The unweighted mean divides by every element: |diff| is [1,0,3,0].
	plain := L1Loss(prediction, target, LossOptions{})
	defer plain.ReleaseGraph()
	value, err = plain.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "l1 mean", value, []float32{1}, 1e-7, 1e-6)
}

func TestLossValidation(t *testing.T) {
	prediction := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CPU, true)
	target := conv2dTestTensor(t, []float32{1, 1}, []int{2}, tensor.CPU, false)
	mismatched := conv2dTestTensor(t, []float32{1, 2, 3}, []int{3}, tensor.CPU, false)
	expectPanic := func(name string, run func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s case did not panic", name)
			}
		}()
		run()
	}
	expectPanic("shape", func() { MSELoss(prediction, mismatched, LossOptions{}) })
	expectPanic("weight shape", func() { L1Loss(prediction, target, LossOptions{Weight: mismatched}) })
	expectPanic("reduction", func() { MSELoss(prediction, target, LossOptions{Reduction: Reduction(7)}) })
	// A scalar elementwise loss works: reducing over no axes returns the value.
	scalar := conv2dTestTensor(t, []float32{3}, []int{}, tensor.CPU, true)
	zero := conv2dTestTensor(t, []float32{0}, []int{}, tensor.CPU, false)
	loss := MSELoss(scalar, zero, LossOptions{})
	defer loss.ReleaseGraph()
	value, err := loss.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "scalar mse", value, []float32{9}, 0, 0)
}

// TestLossRejectsMixedDevices covers the device check, which needs both backends.
func TestLossRejectsMixedDevices(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	pool2dContext(t, tensor.CUDA)
	prediction := conv2dTestTensor(t, []float32{1, 2}, []int{2}, tensor.CUDA, true)
	target := conv2dTestTensor(t, []float32{1, 1}, []int{2}, tensor.CPU, false)
	defer func() {
		if recover() == nil {
			t.Fatal("mixed-device loss did not panic")
		}
	}()
	MSELoss(prediction, target, LossOptions{})
}
