package nn

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// A prior output is caller-owned after Forward, so feeding it back as the next
// input must not be freed by the automatic cleanup at the start of Forward.
func auditSequentialReuse(t *testing.T, device tensor.Device, blas *cuda.Blas) {
	t.Helper()
	auditDeviceThread(t, device)
	start := cuda.MemoryStats().LiveBytes
	model := &Sequential{Modules: []Module{&Tanh{}}}
	x := tensorOn(t, device, []int{1, 3}, []float32{-.5, .2, 1.1})
	first, err := model.Forward(blas, x)
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.Forward(blas, first)
	if err != nil {
		t.Fatalf("previous output was not usable as input: %v", err)
	}
	want := make([]float32, 3)
	for i, v := range host(t, x) {
		want[i] = float32(math.Tanh(math.Tanh(float64(v))))
	}
	auditCompare(t, "chained tanh", host(t, second), want)
	first.Close()
	second.Close()
	model.CloseActivations()
	x.Close()
	auditLiveBytes(t, device, start)
}

func TestAuditSequentialReuseCPU(t *testing.T) { auditSequentialReuse(t, tensor.CPU, nil) }

func TestAuditSequentialReuseCUDA(t *testing.T) {
	auditSequentialReuse(t, tensor.CUDA, auditCUDA(t))
}
