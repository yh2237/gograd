package autograd

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// Separate entry points keep CPU-only validation from touching the CUDA driver.
func TestAuditLossShapeCPU(t *testing.T) { auditLossShapeRegressions(t, tensor.CPU) }

func TestAuditLossShapeCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	auditLossShapeRegressions(t, tensor.CUDA)
}

func auditLossShapeRegressions(t *testing.T, device tensor.Device) {
	run := func(name string, fn func(*testing.T)) {
		t.Run(name, func(t *testing.T) {
			pool2dContext(t, device)
			fn(t)
		})
	}
	run("CrossEntropyEmptyBatch", func(t *testing.T) {
		x := conv2dTestTensor(t, nil, []int{0, 3}, device, true)
		y := CrossEntropy(x, nil, -1)
		defer y.ReleaseGraph()
		v := conv2dHost(t, y, false)
		if len(v) != 1 || !math.IsNaN(float64(v[0])) {
			t.Fatalf("empty cross entropy=%v, want NaN", v)
		}
		if err := y.Backward(); err != nil {
			t.Fatal(err)
		}
		if x.Grad != nil && len(x.Grad) != 0 {
			t.Fatalf("empty cross entropy produced %d gradients", len(x.Grad))
		}
	})
	run("BCEStableZeroAndDifferentiableWeightsTargets", func(t *testing.T) {
		for _, reduction := range []Reduction{ReductionMean, ReductionSum} {
			t.Run(reduction.String(), func(t *testing.T) {
				pool2dContext(t, device)
				xs := []float32{0, 0, -100, 100, -1e-6, 1e-6, -2, 2}
				ts := []float32{0, 1, 1, 0, .2, .8, .3, .7}
				ws := []float32{1, 2, 0, .5, 1, 3, 2, 1}
				x := conv2dTestTensor(t, xs, []int{8}, device, true)
				y := conv2dTestTensor(t, ts, []int{8}, device, true)
				w := conv2dTestTensor(t, ws, []int{8}, device, true)
				loss := BCEWithLogitsLoss(x, y, LossOptions{Reduction: reduction, Weight: w})
				defer loss.ReleaseGraph()
				var want float32
				dx, dt, dw := make([]float32, 8), make([]float32, 8), make([]float32, 8)
				scale := float32(1)
				if reduction == ReductionMean {
					scale /= 8
				}
				for i, value := range xs {
					v := float64(value)
					bce := float32(math.Max(v, 0) - v*float64(ts[i]) + math.Log1p(math.Exp(-math.Abs(v))))
					want += bce * ws[i] * scale
					dx[i] = (float32(1/(1+math.Exp(-v))) - ts[i]) * ws[i] * scale
					dt[i] = -value * ws[i] * scale
					dw[i] = bce * scale
				}
				conv2dClose(t, "loss", conv2dHost(t, loss, false), []float32{want}, 2e-5, 2e-6)
				if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, "dx", conv2dHost(t, x, true), dx, 2e-6, 2e-6)
				conv2dClose(t, "dt", conv2dHost(t, y, true), dt, 2e-6, 2e-6)
				conv2dClose(t, "dw", conv2dHost(t, w, true), dw, 2e-6, 2e-6)
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				for i := range dx {
					dx[i] *= 2
					dt[i] *= 2
					dw[i] *= 2
				}
				conv2dClose(t, "retained dx", conv2dHost(t, x, true), dx, 3e-6, 2e-6)
				conv2dClose(t, "retained dt", conv2dHost(t, y, true), dt, 3e-6, 2e-6)
				conv2dClose(t, "retained dw", conv2dHost(t, w, true), dw, 3e-6, 2e-6)
			})
		}
	})
	run("BCEFrozenAndAliasedOperands", func(t *testing.T) {
		for _, flags := range [][2]bool{{true, false}, {false, true}, {false, false}} {
			x := conv2dTestTensor(t, []float32{0, 2}, []int{2}, device, flags[0])
			y := conv2dTestTensor(t, []float32{1, .25}, []int{2}, device, flags[1])
			loss := BCEWithLogitsLoss(x, y, LossOptions{Reduction: ReductionSum})
			if flags[0] || flags[1] {
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
			} else if loss.RequiresGrad {
				t.Fatal("frozen loss recorded gradients")
			}
			if flags[0] {
				conv2dClose(t, "only dx", conv2dHost(t, x, true), []float32{-.5, float32(1/(1+math.Exp(-2))) - .25}, 1e-6, 1e-6)
			}
			if flags[1] {
				conv2dClose(t, "only dt", conv2dHost(t, y, true), []float32{0, -2}, 1e-6, 1e-6)
			}
			loss.ReleaseGraph()
		}
		x := conv2dTestTensor(t, []float32{0, .25}, []int{2}, device, true)
		loss := Add(BCEWithLogitsLoss(x, x, LossOptions{Reduction: ReductionSum}), Sum(x, 0))
		defer loss.ReleaseGraph()
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "aliased accumulated dx", conv2dHost(t, x, true), []float32{1.5, 1 + float32(1/(1+math.Exp(-.25))) - .5}, 1e-6, 1e-6)
	})
	run("BCEBroadcastViewsAndNoGrad", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{0, 1, 2, 3}, []int{2, 2}, device, true)
		y := conv2dTestTensor(t, []float32{.25, .75}, []int{2}, device, true)
		loss := BCEWithLogitsLoss(Transpose(x, 0, 1), y, LossOptions{})
		defer loss.ReleaseGraph()
		if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "broadcast dt", conv2dHost(t, y, true), []float32{-.25, -1.25}, 1e-6, 1e-6)
		want := []float32{(.5 - .25) / 4, (float32(1/(1+math.Exp(-1))) - .25) / 4, (float32(1/(1+math.Exp(-2))) - .75) / 4, (float32(1/(1+math.Exp(-3))) - .75) / 4}
		conv2dClose(t, "view dx", conv2dHost(t, x, true), want, 1e-6, 1e-6)
		NoGrad(func() {
			out := BCEWithLogitsLoss(x, y, LossOptions{})
			defer out.ReleaseGraph()
			if out.RequiresGrad || out.backward != nil || out.backwardGPU != nil {
				t.Fatal("NoGrad recorded BCE history")
			}
		})
	})
	run("BCESavedValueVersions", func(t *testing.T) {
		for mutate := 0; mutate < 3; mutate++ {
			x := conv2dTestTensor(t, []float32{0, 1}, []int{2}, device, true)
			y := conv2dTestTensor(t, []float32{.25, .75}, []int{2}, device, false)
			w := conv2dTestTensor(t, []float32{1, 1}, []int{2}, device, false)
			loss := BCEWithLogitsLoss(x, y, LossOptions{Weight: w})
			if err := []*Tensor{x, y, w}[mutate].CopyFrom([]float32{.5, .5}); err != nil {
				t.Fatal(err)
			}
			if err := loss.Backward(); err == nil || !strings.Contains(err.Error(), "modified") {
				t.Fatalf("mutation %d was not rejected: %v", mutate, err)
			}
			loss.ReleaseGraph()
		}
	})
	run("BCEBroadcastLogitsAndWeights", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{1}, []int{1}, device, true)
		y := conv2dTestTensor(t, []float32{0, 0, .25, .75}, []int{2, 2}, device, true)
		w := conv2dTestTensor(t, []float32{2}, nil, device, true)
		loss := BCEWithLogitsLoss(x, y, LossOptions{Weight: w})
		defer loss.ReleaseGraph()
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "broadcast logits dx", conv2dHost(t, x, true), []float32{2 * (float32(1/(1+math.Exp(-1))) - .25)}, 1e-6, 1e-6)
		conv2dClose(t, "broadcast logits dt", conv2dHost(t, y, true), []float32{-.5, -.5, -.5, -.5}, 1e-6, 1e-6)
		conv2dClose(t, "broadcast weights dw", conv2dHost(t, w, true), []float32{float32(math.Log1p(math.Exp(1))) - .25}, 1e-6, 1e-6)
	})
	run("BCEEmptyAndScalar", func(t *testing.T) {
		x := conv2dTestTensor(t, nil, []int{0}, device, true)
		y := conv2dTestTensor(t, nil, []int{0}, device, false)
		for _, r := range []Reduction{ReductionSum, ReductionMean} {
			loss := BCEWithLogitsLoss(x, y, LossOptions{Reduction: r})
			value := conv2dHost(t, loss, false)[0]
			// Preserve each reduction backend's pre-existing empty semantics.
			if r == ReductionMean && device == tensor.CPU {
				if !math.IsNaN(float64(value)) {
					t.Fatalf("empty CPU mean = %v, want NaN", value)
				}
			} else if value != 0 {
				t.Fatalf("empty reduction = %v, want 0", value)
			}
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			loss.ReleaseGraph()
		}
		s := conv2dTestTensor(t, []float32{0}, nil, device, true)
		z := conv2dTestTensor(t, []float32{1}, nil, device, false)
		loss := BCEWithLogitsLoss(s, z, LossOptions{})
		defer loss.ReleaseGraph()
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "scalar zero dx", conv2dHost(t, s, true), []float32{-.5}, 0, 0)
	})
	run("CrossEntropyLargeShiftAndTargetSnapshot", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{1e8, 1e8, 1e8, 1e8, 1e8, 1e8}, []int{3, 2}, device, true)
		ids := []int{0, 1, -1}
		loss := CrossEntropy(x, ids, -1)
		defer loss.ReleaseGraph()
		ids[0], ids[1], ids[2] = -1, 999, 0
		conv2dClose(t, "shifted CE", conv2dHost(t, loss, false), []float32{float32(math.Log(2))}, 1e-6, 1e-6)
		for pass := 1; pass <= 2; pass++ {
			if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: pass == 1}); err != nil {
				t.Fatal(err)
			}
			s := float32(pass) * .25
			conv2dClose(t, "snapshot CE dx", conv2dHost(t, x, true), []float32{-s, s, s, -s, 0, 0}, 1e-6, 1e-6)
		}
	})
	run("EmbeddingHostMetadataSnapshot", func(t *testing.T) {
		w := conv2dTestTensor(t, []float32{0, 1, 2, 3, 4, 5}, []int{3, 2}, device, true)
		ids, shape := []int{1, 1, 0}, []int{3}
		y := EmbeddingWithPadding(w, ids, shape, 0)
		loss := Sum(y, 0, 1)
		defer loss.ReleaseGraph()
		ids[0], ids[1], ids[2], shape[0] = 2, 999, 2, 999
		if !slices.Equal(y.Shape, []int{3, 2}) {
			t.Fatalf("embedding shape changed: %v", y.Shape)
		}
		for pass := 1; pass <= 2; pass++ {
			if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: pass == 1}); err != nil {
				t.Fatal(err)
			}
			conv2dClose(t, "embedding snapshot", conv2dHost(t, w, true), []float32{0, 0, float32(2 * pass), float32(2 * pass), 0, 0}, 0, 0)
		}
	})
	run("ConcatOccurrencesAndCallerSlice", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{1, 2}, []int{2, 1}, device, true)
		frozen := conv2dTestTensor(t, []float32{3, 4}, []int{2, 1}, device, false)
		view := Transpose(x, 0, 1)
		inputs := []*Tensor{view, view}
		out := Concat(0, inputs...)
		defer out.ReleaseGraph()
		if inputs[0] != view || inputs[1] != view {
			t.Fatal("Concat rewrote the caller's input slice")
		}
		inputs[0], inputs[1] = frozen, frozen
		weights := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, false)
		y := Concat(-1, x, x, frozen)
		loss := Sum(Mul(y, weights), 0, 1)
		defer loss.ReleaseGraph()
		for pass := 1; pass <= 2; pass++ {
			if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: pass == 1}); err != nil {
				t.Fatal(err)
			}
			conv2dClose(t, "concat occurrence VJP", conv2dHost(t, x, true), []float32{float32(3 * pass), float32(9 * pass)}, 0, 0)
		}
		if frozen.Grad != nil || frozen.gradBuf != nil {
			t.Fatal("Concat allocated a frozen-input gradient")
		}
	})
	run("ConcatCapturedSliceAndInvalidAxis", func(t *testing.T) {
		x := conv2dTestTensor(t, []float32{1}, []int{1}, device, true)
		y := conv2dTestTensor(t, []float32{2}, []int{1}, device, true)
		inputs := []*Tensor{x, y}
		loss := Sum(Concat(0, inputs...), 0)
		defer loss.ReleaseGraph()
		inputs[0], inputs[1] = y, y
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "captured concat x", conv2dHost(t, x, true), []float32{1}, 0, 0)
		conv2dClose(t, "captured concat y", conv2dHost(t, y, true), []float32{1}, 0, 0)
		for _, axis := range []int{-2, 1, 99} {
			func() {
				defer func() {
					p := recover()
					if p != "autograd: invalid concat axis" {
						t.Errorf("axis %d: expected clean concat axis error, got %v", axis, p)
					}
				}()
				Concat(axis, x, y)
			}()
		}
	})
}

func TestAuditBCECompactGraphCPU(t *testing.T) {
	x := conv2dTestTensor(t, []float32{0, 1}, []int{2}, tensor.CPU, true)
	y := conv2dTestTensor(t, []float32{0, 1}, []int{2}, tensor.CPU, true)
	loss := BCEWithLogitsLoss(x, y, LossOptions{})
	defer loss.ReleaseGraph()
	if len(loss.parents) != 1 || len(loss.parents[0].parents) != 2 || loss.parents[0].parents[0] != x || loss.parents[0].parents[1] != y {
		t.Fatal("BCE should have only an elementwise node and a reduction")
	}
}
