package autograd

import (
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func BenchmarkAuditBCECPU(b *testing.B) { benchmarkAuditBCE(b, tensor.CPU) }

func BenchmarkAuditBCECUDA(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Close()
	benchmarkAuditBCE(b, tensor.CUDA)
}

func benchmarkAuditBCE(b *testing.B, device tensor.Device) {
	x, err := Zeros([]int{4096}, device, true)
	if err != nil {
		b.Fatal(err)
	}
	defer x.Close()
	y, err := Zeros([]int{4096}, device, false)
	if err != nil {
		b.Fatal(err)
	}
	defer y.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.ZeroGrad()
		loss := BCEWithLogitsLoss(x, y, LossOptions{})
		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}
		loss.ReleaseGraph()
	}
	if device == tensor.CUDA {
		if _, err := x.GradToHost(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAuditCrossEntropyCPU(b *testing.B) {
	x := Must(make([]float32, 32*1024), []int{32, 1024}, true)
	defer x.Close()
	ids := make([]int, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.ZeroGrad()
		loss := CrossEntropy(x, ids, -1)
		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}
		loss.ReleaseGraph()
	}
}

func BenchmarkAuditConcatManyCPU(b *testing.B) {
	x := Must(make([]float32, 16), []int{16}, true)
	defer x.Close()
	frozen := Must(make([]float32, 16), []int{16}, false)
	defer frozen.Close()
	inputs := make([]*Tensor, 256)
	for i := range inputs {
		inputs[i] = frozen
		if i%2 == 0 {
			inputs[i] = x
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.ZeroGrad()
		loss := Sum(Concat(0, inputs...), 0)
		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}
		loss.ReleaseGraph()
	}
}
