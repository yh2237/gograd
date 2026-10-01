package nn

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestMLPTrainingReducesLoss(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	blas, err := cuda.NewBlas()
	if err != nil {
		t.Fatal(err)
	}
	defer blas.Destroy()

	rng := rand.New(rand.NewSource(5))
	const rows, inputs, hidden = 64, 2, 16
	xHost := make([]float32, rows*inputs)
	yHost := make([]float32, rows)
	for i := 0; i < rows; i++ {
		a := rng.NormFloat64()
		b := rng.NormFloat64()
		xHost[i*inputs] = float32(a)
		xHost[i*inputs+1] = float32(b)
		yHost[i] = float32(math.Sin(a) + 0.5*b)
	}
	x, err := tensor.FromHost([]int{rows, inputs}, xHost)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()

	layer1, err := NewLinear(inputs, hidden, rng)
	if err != nil {
		t.Fatal(err)
	}
	layer2, err := NewLinear(hidden, 1, rng)
	if err != nil {
		t.Fatal(err)
	}
	model := &Sequential{Modules: []Module{layer1, &Tanh{}, layer2}}
	defer model.Close()

	optimizer, err := NewAdamW(model.Parameters(), model.Gradients(), 0.01, 0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	defer optimizer.Close()

	evaluate := func() float64 {
		out, err := model.Forward(blas, x)
		if err != nil {
			t.Fatal(err)
		}
		predicted, err := out.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		loss, _ := MSE(predicted, yHost)
		model.CloseActivations()
		return loss
	}

	initial := evaluate()
	for step := 0; step < 300; step++ {
		model.ZeroGrad()
		out, err := model.Forward(blas, x)
		if err != nil {
			t.Fatal(err)
		}
		predicted, err := out.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		_, gradient := MSE(predicted, yHost)
		gradTensor, err := tensor.FromHost([]int{rows, 1}, gradient)
		if err != nil {
			t.Fatal(err)
		}
		dx, err := model.Backward(blas, gradTensor)
		if err != nil {
			t.Fatal(err)
		}
		dx.Close()
		gradTensor.Close()
		model.CloseActivations()
		if err := optimizer.Step(); err != nil {
			t.Fatal(err)
		}
	}
	final := evaluate()
	t.Logf("MLP initial loss %.6f -> final loss %.6f", initial, final)
	if !(final < initial*0.5) {
		t.Errorf("training did not reduce loss: initial %.6f final %.6f", initial, final)
	}
}
