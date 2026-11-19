package autograd

import (
	"math"
	"strings"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestAuditMatMulSlicedUnitColumns(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		pool2dContext(t, device)
		a := conv2dTestTensor(t, []float32{2, 3, 4}, []int{3, 1}, device, true)
		w := conv2dTestTensor(t, []float32{5, 7, 11, 13}, []int{4, 1}, device, true)
		b := Slice(w, 0, 0, 1)
		y := MatMul(a, b)
		loss := Sum(y, 0, 1)
		defer loss.ReleaseGraph()
		conv2dClose(t, "2d out", conv2dHost(t, y, false), []float32{10, 15, 20}, 1e-6, 1e-6)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "2d da", conv2dHost(t, a, true), []float32{5, 5, 5}, 1e-6, 1e-6)
		conv2dClose(t, "2d dw", conv2dHost(t, w, true), []float32{9, 0, 0, 0}, 1e-6, 1e-6)

		a3 := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2, 1}, device, true)
		w3 := conv2dTestTensor(t, []float32{2, 3, 5, 7}, []int{4, 1, 1}, device, true)
		b3 := Slice(w3, 0, 0, 2)
		y3 := MatMul(a3, b3)
		loss3 := Sum(y3, 0, 1, 2)
		defer loss3.ReleaseGraph()
		conv2dClose(t, "3d out", conv2dHost(t, y3, false), []float32{2, 4, 9, 12}, 1e-6, 1e-6)
		if err := loss3.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "3d da", conv2dHost(t, a3, true), []float32{2, 2, 3, 3}, 1e-6, 1e-6)
		conv2dClose(t, "3d dw", conv2dHost(t, w3, true), []float32{3, 7, 0, 0}, 1e-6, 1e-6)
	})
}

func TestAuditSmoothL1LargeAndEdges(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		pool2dContext(t, device)
		xs := []float32{-1e20, -1e8, -1e4, -1, -.5, 0, .5, 1, 1e4, 1e8, 1e20}
		ys := make([]float32, len(xs))
		x := conv2dTestTensor(t, xs, []int{len(xs)}, device, true)
		y := conv2dTestTensor(t, ys, []int{len(ys)}, device, true)
		loss := SmoothL1Loss(x, y, LossOptions{Reduction: ReductionSum})
		defer loss.ReleaseGraph()
		got := conv2dHost(t, loss, false)[0]
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || got <= 0 {
			t.Fatalf("large SmoothL1Loss = %g, want finite positive", got)
		}
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		wantGrad := []float32{-1, -1, -1, -1, -.5, 0, .5, 1, 1, 1, 1}
		wantNeg := make([]float32, len(wantGrad))
		for i, v := range wantGrad {
			wantNeg[i] = -v
		}
		conv2dClose(t, "dx", conv2dHost(t, x, true), wantGrad, 1e-6, 1e-6)
		conv2dClose(t, "dt", conv2dHost(t, y, true), wantNeg, 1e-6, 1e-6)

		bx := conv2dTestTensor(t, []float32{1e8, -2}, []int{2, 1}, device, true)
		by := conv2dTestTensor(t, []float32{0, 1}, []int{1, 2}, device, true)
		bw := conv2dTestTensor(t, []float32{1, 2, 0, 1}, []int{2, 2}, device, false)
		bloss := SmoothL1Loss(bx, by, LossOptions{Weight: bw})
		defer bloss.ReleaseGraph()
		if err := bloss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "broadcast dx", conv2dHost(t, bx, true), []float32{.75, -.25}, 1e-6, 1e-6)
		conv2dClose(t, "broadcast dt", conv2dHost(t, by, true), []float32{-.25, -.25}, 1e-6, 1e-6)
		if err := bloss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "retained dx", conv2dHost(t, bx, true), []float32{1.5, -.5}, 1e-6, 1e-6)

		ax := conv2dTestTensor(t, []float32{1, 2}, []int{2}, device, true)
		aloss := SmoothL1Loss(ax, ax, LossOptions{Reduction: ReductionSum})
		defer aloss.ReleaseGraph()
		conv2dClose(t, "aliased loss", conv2dHost(t, aloss, false), []float32{0}, 0, 0)
		if err := aloss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "aliased dx", conv2dHost(t, ax, true), []float32{0, 0}, 0, 0)

		vx := conv2dTestTensor(t, []float32{1, 2}, []int{2}, device, true)
		vy := conv2dTestTensor(t, []float32{0, 0}, []int{2}, device, false)
		vloss := SmoothL1Loss(vx, vy, LossOptions{})
		defer vloss.ReleaseGraph()
		if err := vx.CopyFrom([]float32{3, 4}); err != nil {
			t.Fatal(err)
		}
		if err := vloss.Backward(); err == nil || !strings.Contains(err.Error(), "modified") {
			t.Fatalf("expected version error, got %v", err)
		}

		empty4d := conv2dTestTensor(t, nil, []int{0, 2, 3, 4}, device, false)
		m4d := Mean(empty4d, 2)
		defer m4d.ReleaseGraph()
		if len(conv2dHost(t, m4d, false)) != 0 {
			t.Fatal("empty 4D mean returned non-empty output")
		}

		ep := conv2dTestTensor(t, nil, []int{0, 2}, device, false)
		et := conv2dTestTensor(t, []float32{1, 2}, []int{1, 2}, device, true)
		eloss := SmoothL1Loss(ep, et, LossOptions{Reduction: ReductionSum})
		defer eloss.ReleaseGraph()
		if err := eloss.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "empty smooth_l1 broadcast dt", conv2dHost(t, et, true), []float32{0, 0}, 0, 0)
	})
}

func TestAuditSwiGLUAndSigmoidLargeInputs(t *testing.T) {
	pool2dDevices(t, func(t *testing.T, device tensor.Device) {
		pool2dContext(t, device)
		x := conv2dTestTensor(t, []float32{-200, -100, -50, 0, 50, 100, 200}, []int{1, 7}, device, true)
		w1 := conv2dTestTensor(t, []float32{
			1, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 1,
		}, []int{3, 7}, device, true)
		w3 := conv2dTestTensor(t, []float32{
			1, 1, 1, 1, 1, 1, 1,
			1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 1,
		}, []int{3, 7}, device, true)
		w2 := conv2dTestTensor(t, []float32{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
			1, 1, 1,
			0, 0, 1,
			0, 1, 0,
			1, 0, 0,
		}, []int{7, 3}, device, true)
		out := SwiGLUProjection(x, w1, w2, w3)
		loss := Sum(out, 0, 1)
		defer loss.ReleaseGraph()
		for i, v := range conv2dHost(t, out, false) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("SwiGLU out[%d] = %g", i, v)
			}
		}
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		for name, p := range map[string]*Tensor{"x": x, "w1": w1, "w2": w2, "w3": w3} {
			for i, v := range conv2dHost(t, p, true) {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("SwiGLU grad %s[%d] = %g", name, i, v)
				}
			}
		}

		sx := conv2dTestTensor(t, []float32{-200, -100, 0, 100, 200}, []int{5}, device, true)
		sy := Add(Sigmoid(sx), Softplus(sx))
		sloss := Sum(sy, 0)
		defer sloss.ReleaseGraph()
		if err := sloss.Backward(); err != nil {
			t.Fatal(err)
		}
		for i, v := range conv2dHost(t, sx, true) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("sigmoid+softplus grad[%d] = %g", i, v)
			}
		}
	})
}

func TestAuditTransformerTrainPropagation(t *testing.T) {
	enc, err := NewTransformerEncoderLayer(8, 2, 16, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, p := range enc.NamedParameters() {
			p.Value.Close()
		}
	}()
	if !enc.Training || !enc.Attention.Training {
		t.Fatal("expected encoder and attention to start in training mode")
	}
	enc.Train(false)
	if enc.Training || enc.Attention.Training {
		t.Fatalf("Train(false) did not propagate: enc=%v attn=%v", enc.Training, enc.Attention.Training)
	}
	enc.Train(true)
	if !enc.Training || !enc.Attention.Training {
		t.Fatalf("Train(true) did not propagate: enc=%v attn=%v", enc.Training, enc.Attention.Training)
	}
	if _, err := NewTransformerEncoderLayer(8, 2, -1, tensor.CPU); err == nil {
		t.Fatal("expected error for negative feedForward")
	}
}

func TestAuditCPUCopyFromRejectsOverlap(t *testing.T) {
	base := Must([]float32{1, 2}, []int{1, 2}, false)
	defer base.Close()
	expanded := Expand(base, 3, 2)
	defer expanded.Close()
	beforeVer := base.storage.version.Load()
	if err := expanded.CopyFrom([]float32{1, 2, 3, 4, 5, 6}); err == nil {
		t.Fatal("CopyFrom accepted overlapping Expand view")
	}
	if base.storage.version.Load() != beforeVer {
		t.Fatal("failed CopyFrom bumped storage version")
	}
	got, err := base.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "base unchanged", got, []float32{1, 2}, 0, 0)

	mat := Must([]float32{1, 2, 3, 4}, []int{2, 2}, false)
	defer mat.Close()
	tr := Transpose(mat, 0, 1)
	defer tr.Close()
	if err := tr.CopyFrom([]float32{10, 20, 30, 40}); err != nil {
		t.Fatalf("non-overlapping Transpose CopyFrom failed: %v", err)
	}
	got, err = mat.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "transposed write", got, []float32{10, 30, 20, 40}, 0, 0)

	selfMat := Must([]float32{1, 2, 3, 4}, []int{2, 2}, false)
	defer selfMat.Close()
	selfTr := Transpose(selfMat, 0, 1)
	defer selfTr.Close()
	if err := selfTr.CopyFrom(selfMat.Data); err != nil {
		t.Fatalf("self-aliased Transpose CopyFrom failed: %v", err)
	}
	got, err = selfMat.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "self-aliased transpose write", got, []float32{1, 3, 2, 4}, 0, 0)
}

func TestAuditCPUPoolBuckets(t *testing.T) {
	sizes := []int{0, 1, 64, 65, 80, 96, 128, 129, 255, 256, 4097, maxCPUPoolSize}
	for _, n := range sizes {
		buf := cpuAllocUninit(n)
		if len(buf) != n {
			t.Fatalf("cpuAllocUninit(%d): len=%d", n, len(buf))
		}
		if n > 0 {
			idx, size := cpuPoolBucket(n)
			if idx < 0 || idx >= len(cpuPools) || size < n || cap(buf) != size {
				t.Fatalf("bucket(%d) = (%d, %d), cap=%d", n, idx, size, cap(buf))
			}
			idx2, size2 := cpuPoolBucket(size)
			if idx2 != idx || size2 != size {
				t.Fatalf("bucket roundtrip(%d): (%d,%d) != (%d,%d)", size, idx2, size2, idx, size)
			}
		}
		cpuRelease(buf)
	}
	cpuRelease(make([]float32, 37))
}
