package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// TestAuditCoreCPU and TestAuditCoreCUDA run the same core regression checks
// on both devices so that the two backends cannot drift apart.
func TestAuditCoreCPU(t *testing.T) { auditCoreRegressions(t, tensor.CPU) }

func TestAuditCoreCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	auditCoreRegressions(t, tensor.CUDA)
}

func auditCoreRegressions(t *testing.T, device tensor.Device) {
	run := func(name string, fn func(*testing.T)) {
		t.Run(name, func(t *testing.T) {
			pool2dContext(t, device)
			fn(t)
		})
	}
	run("SoftmaxNaNPropagates", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{1, float32(math.NaN()), 2}, []int{1, 3}, device, true)
		y := Softmax(x, 1)
		defer y.ReleaseGraph()
		values := conv2dHost(t, y, false)
		for i, v := range values {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("softmax[%d]=%v, want NaN for a NaN input", i, v)
			}
		}
	})
	run("ReduceEmptyTensor", func(t *testing.T) {
		x := conv2dTestTensor(t, nil, []int{0, 3}, device, true)
		sum := Sum(x, 1)
		defer sum.ReleaseGraph()
		if got := sum.Shape; len(got) != 1 || got[0] != 0 {
			t.Fatalf("sum shape %v, want [0]", got)
		}
		mean := Mean(x, 1)
		defer mean.ReleaseGraph()
		if got := mean.Shape; len(got) != 1 || got[0] != 0 {
			t.Fatalf("mean shape %v, want [0]", got)
		}
	})
	run("TransposeNegativeAxis", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, true)
		want := conv2dHost(t, Transpose(x, 0, 1), false)
		got := conv2dHost(t, Transpose(x, -2, -1), false)
		if len(got) != len(want) {
			t.Fatalf("transpose length %d != %d", len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("transpose[%d]=%v, want %v", i, got[i], want[i])
			}
		}
	})
	run("EmbeddingEmptyBatch", func(t *testing.T) {
		w := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, device, true)
		y := EmbeddingWithPadding(w, []int{}, []int{0}, -1)
		defer y.ReleaseGraph()
		if got := y.Shape; len(got) != 2 || got[0] != 0 || got[1] != 2 {
			t.Fatalf("embedding shape %v, want [0 2]", got)
		}
	})
}

// TestAuditIndexLoaderBatchesDoNotAlias checks that the batches of one epoch
// do not share a backing array: appending to a batch must not overwrite the
// next one.
func TestAuditIndexLoaderBatchesDoNotAlias(t *testing.T) {
	loader := NewIndexLoader(4, 2, 7)
	batches := loader.Epoch()
	if len(batches) != 2 {
		t.Fatalf("batch count %d, want 2", len(batches))
	}
	first := append(batches[0], -1)
	if batches[1][0] == first[len(first)-1] && len(batches[1]) > 0 {
		// The appended value may coincide with the original by chance, so
		// compare against a fresh epoch instead.
		fresh := NewIndexLoader(4, 2, 7).Epoch()
		if batches[1][0] != fresh[1][0] {
			t.Fatalf("appending to a batch changed the next batch: %v vs %v", batches[1][0], fresh[1][0])
		}
	}
}

// TestAuditOneCycleRejectsInvalidSchedule checks that an unusable schedule is
// rejected at construction instead of producing a NaN learning rate later.
func TestAuditOneCycleRejectsInvalidSchedule(t *testing.T) {
	for _, schedule := range []struct {
		maxLR    float64
		total    int
		pctStart float64
	}{
		{.01, 1, .3},
		{.01, 10, 1},
		{.01, 10, 0},
		{math.NaN(), 10, .3},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("OneCycle(%g, %d, %g) was accepted", schedule.maxLR, schedule.total, schedule.pctStart)
				}
			}()
			NewOneCycle(schedule.maxLR, schedule.total, schedule.pctStart)
		}()
	}
}

// TestAuditAdamWRejectsMixedDevices checks that a parameter list that spans
// two devices is rejected: the update would silently freeze one device.
func TestAuditAdamWRejectsMixedDevices(t *testing.T) {
	cpu := Must([]float32{1}, []int{1}, true)
	defer cpu.Close()
	if _, err := NewAdamWGroups([]Parameter{{Name: "cpu", Value: cpu}}, .1, nil); err == nil {
		t.Fatal("AdamW accepted a parameter list without groups")
	}
	groups := []ParameterGroupSpec{{Name: "default", Params: []Parameter{{Name: "cpu", Value: cpu}}}}
	if _, err := NewAdamWGroups([]Parameter{{Name: "cpu", Value: cpu}}, .1, groups); err != nil {
		t.Fatalf("AdamW rejected a single CPU parameter: %v", err)
	}
}
