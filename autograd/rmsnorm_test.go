package autograd

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestRMSNormLastIrodoriParity(t *testing.T) {
	data, err := os.ReadFile("../testdata/irodori_primitives.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Input  []float32 `json:"rms_input"`
		Weight []float32 `json:"rms_weight"`
		Output []float32 `json:"rms_output"`
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		name := "cpu"
		if device == tensor.CUDA {
			name = "cuda"
		}
		t.Run(name, func(t *testing.T) {
			if device == tensor.CUDA && !cuda.Available() {
				t.Skip("CUDA unavailable")
			}
			if device == tensor.CUDA {
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				defer ctx.Close()
			}
			x, err := New(f.Input, []int{len(f.Input) / 4, 4}, device, false)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			w, err := New(f.Weight, []int{4}, device, false)
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()
			y := RMSNormLast(x, w, 1e-6)
			defer y.Close()
			got, err := y.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			var maxAbs float64
			for i, v := range got {
				maxAbs = max(maxAbs, math.Abs(float64(v-f.Output[i])))
			}
			t.Logf("max_abs=%.9g", maxAbs)
			if maxAbs > 3e-6 {
				t.Fatalf("max abs %.9g", maxAbs)
			}
		})
	}
}

func BenchmarkRMSNormLast(b *testing.B) {
	xdata := make([]float32, 256*1024)
	wdata := make([]float32, 1024)
	for i := range wdata {
		wdata[i] = 1
	}
	x, _ := New(xdata, []int{256, 1024}, tensor.CPU, false)
	defer x.Close()
	w, _ := New(wdata, []int{1024}, tensor.CPU, false)
	defer w.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y := RMSNormLast(x, w, 1e-6)
		y.Close()
	}
}

func BenchmarkRMSNormLastCUDA(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Close()
	xdata := make([]float32, 256*1024)
	wdata := make([]float32, 1024)
	for i := range wdata {
		wdata[i] = 1
	}
	x, err := New(xdata, []int{256, 1024}, tensor.CUDA, false)
	if err != nil {
		b.Fatal(err)
	}
	defer x.Close()
	w, err := New(wdata, []int{1024}, tensor.CUDA, false)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y := RMSNormLast(x, w, 1e-6)
		if _, err := y.ToHost(); err != nil {
			b.Fatal(err)
		}
		y.Close()
	}
}
