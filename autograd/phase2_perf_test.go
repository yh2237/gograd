package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func seqData(n int, scale, phase float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i)*0.17+phase) * scale)
	}
	return out
}

func mustNew(t *testing.T, data []float32, shape []int, dev tensor.Device, grad bool) *Tensor {
	t.Helper()
	x, err := New(data, shape, dev, grad)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func mustHost(t *testing.T, x *Tensor) []float32 {
	t.Helper()
	out, err := x.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func mustGradHost(t *testing.T, x *Tensor) []float32 {
	t.Helper()
	out, err := x.GradToHost()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func assertCloseSlice(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", label, len(got), len(want))
	}
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("%s[%d]: got %g want NaN", label, i, got[i])
			}
			continue
		}
		if math.IsNaN(g) || !(math.Abs(g-w) <= tol) {
			t.Fatalf("%s[%d]: got %g want %g (diff %g > %g)", label, i, got[i], want[i], math.Abs(g-w), tol)
		}
	}
}

func TestFlashAttentionEdgeCasesCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	defer func() { AttentionAlgorithm = "auto" }()

	cases := []struct {
		name          string
		batch         []int
		nq, nk, d, e  int
		withMask      bool
		reqQ, reqK    bool
		reqV, reqMask bool
		aliasQK       bool
		aliasKV       bool
		aliasQKV      bool
	}{
		{name: "d128_e128", batch: []int{1, 2}, nq: 35, nk: 41, d: 128, e: 128, withMask: true, reqQ: true, reqK: true, reqV: true, reqMask: true},
		{name: "d96_e80_odd", batch: []int{2, 1}, nq: 19, nk: 37, d: 96, e: 80, withMask: true, reqQ: true, reqK: true, reqV: true},
		{name: "d63_e63", batch: []int{1, 2}, nq: 33, nk: 65, d: 63, e: 63, withMask: true, reqQ: true, reqK: true, reqV: true, reqMask: true},
		{name: "small_odd", batch: []int{2, 3}, nq: 7, nk: 11, d: 17, e: 23, withMask: true, reqQ: true, reqK: true, reqV: true, reqMask: true},
		{name: "only_q_grad", batch: []int{1, 2}, nq: 33, nk: 65, d: 64, e: 64, reqQ: true},
		{name: "only_kv_grad", batch: []int{1, 2}, nq: 33, nk: 65, d: 64, e: 64, reqK: true, reqV: true},
		{name: "only_mask_grad", batch: []int{1, 2}, nq: 17, nk: 33, d: 32, e: 32, withMask: true, reqMask: true},
		{name: "alias_qk", batch: []int{1, 2}, nq: 21, nk: 21, d: 48, e: 32, reqQ: true, reqK: true, reqV: true, aliasQK: true},
		{name: "alias_kv", batch: []int{1, 2}, nq: 21, nk: 21, d: 48, e: 48, reqQ: true, reqK: true, reqV: true, aliasKV: true},
		{name: "alias_qkv", batch: []int{1, 2}, nq: 21, nk: 21, d: 64, e: 64, reqQ: true, reqK: true, reqV: true, aliasQKV: true},
		{name: "empty_nq", batch: []int{1, 2}, nq: 0, nk: 8, d: 16, e: 16, reqQ: true, reqK: true, reqV: true},
		{name: "empty_nk", batch: []int{1, 2}, nq: 8, nk: 0, d: 16, e: 16, reqQ: true, reqK: true, reqV: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := numel(tc.batch)
			qShape := append(append([]int(nil), tc.batch...), tc.nq, tc.d)
			kShape := append(append([]int(nil), tc.batch...), tc.nk, tc.d)
			vShape := append(append([]int(nil), tc.batch...), tc.nk, tc.e)
			oShape := append(append([]int(nil), tc.batch...), tc.nq, tc.e)
			mShape := []int{1, 1, tc.nq, tc.nk}

			qd := seqData(b*tc.nq*tc.d, 0.25, 0.1)
			kd := seqData(b*tc.nk*tc.d, 0.25, 0.5)
			vd := seqData(b*tc.nk*tc.e, 0.25, 0.9)
			md := seqData(tc.nq*tc.nk, 0.15, 1.3)
			ud := seqData(b*tc.nq*tc.e, 0.2, 1.7)

			run := func(mode string) [][]float32 {
				AttentionAlgorithm = mode
				q := mustNew(t, qd, qShape, tensor.CUDA, tc.reqQ)
				var k, v, mask *Tensor
				switch {
				case tc.aliasQKV:
					k, v = q, q
				case tc.aliasQK:
					k = q
					v = mustNew(t, vd, vShape, tensor.CUDA, tc.reqV)
				case tc.aliasKV:
					k = mustNew(t, kd, kShape, tensor.CUDA, tc.reqK)
					v = k
				default:
					k = mustNew(t, kd, kShape, tensor.CUDA, tc.reqK)
					v = mustNew(t, vd, vShape, tensor.CUDA, tc.reqV)
				}
				if tc.withMask {
					mask = mustNew(t, md, mShape, tensor.CUDA, tc.reqMask)
				}
				up := mustNew(t, ud, oShape, tensor.CUDA, false)
				y := ScaledDotProductAttention(q, k, v, mask)
				out := mustHost(t, y)
				loss := weightedSum(y, up)
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				var qg, kg, vg, mg []float32
				if tc.reqQ {
					qg = mustGradHost(t, q)
				}
				if !tc.aliasQKV && !tc.aliasQK && tc.reqK {
					kg = mustGradHost(t, k)
				}
				if !tc.aliasQKV && !tc.aliasKV && tc.reqV {
					vg = mustGradHost(t, v)
				}
				if tc.withMask && tc.reqMask {
					mg = mustGradHost(t, mask)
				}
				loss.ReleaseGraph()
				up.Close()
				if mask != nil {
					mask.Close()
				}
				if !tc.aliasQKV && !tc.aliasKV {
					v.Close()
				}
				if !tc.aliasQKV && !tc.aliasQK {
					k.Close()
				}
				q.Close()
				return [][]float32{out, qg, kg, vg, mg}
			}

			want := run("materialized")
			got := run("flash")
			names := []string{"out", "dq", "dk", "dv", "dmask"}
			for i := range want {
				assertCloseSlice(t, names[i], got[i], want[i], 3e-3)
			}
		})
	}
}

func TestBF16GraphCaptureWarmCopyFromCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	BF16Autocast = true
	defer func() { BF16Autocast = false }()

	x := mustNew(t, seqData(4*8, 0.5, 0.1), []int{4, 8}, tensor.CUDA, false)
	w := mustNew(t, seqData(6*8, 0.5, 0.4), []int{6, 8}, tensor.CUDA, true)
	defer x.Close()
	defer w.Close()

	warm := matMulTransB(x, w)
	_ = mustHost(t, warm)
	warm.ReleaseGraph()
	if x.bf16Storage == nil || w.bf16Storage == nil {
		t.Fatal("expected warm BF16 shadows before capture")
	}

	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	if err := SetBLASStream(stream); err != nil {
		t.Fatal(err)
	}
	defer SetBLASStream(nil)

	var out *Tensor
	graph, err := cuda.Capture(stream, func() error {
		out = matMulTransB(x, w)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	defer out.ReleaseGraph()

	nextX := seqData(4*8, 0.7, 2.3)
	nextW := seqData(6*8, 0.6, 3.1)
	if err := x.CopyFrom(nextX); err != nil {
		t.Fatal(err)
	}
	if err := w.CopyFrom(nextW); err != nil {
		t.Fatal(err)
	}
	if err := graph.Launch(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := mustHost(t, out)

	xc := mustNew(t, nextX, []int{4, 8}, tensor.CPU, false)
	wc := mustNew(t, nextW, []int{6, 8}, tensor.CPU, false)
	defer xc.Close()
	defer wc.Close()
	ref := matMulTransB(xc, wc)
	defer ref.ReleaseGraph()
	assertCloseSlice(t, "bf16 graph replay after CopyFrom", got, ref.Data, 5e-2)
}

func TestNoGradReleasesAuxCUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()
	defer func() {
		AttentionAlgorithm = "auto"
		BF16Autocast = false
	}()
	BF16Autocast = true

	NoGrad(func() {
		q := mustNew(t, seqData(2*2*8*16, 0.2, 0.1), []int{2, 2, 8, 16}, tensor.CUDA, true)
		defer q.Close()
		AttentionAlgorithm = "materialized"
		ym := ScaledDotProductAttention(q, q, q, nil)
		defer ym.Close()
		if len(ym.aux) != 0 || len(ym.savedLeases) != 0 {
			t.Fatalf("materialized NoGrad retained aux=%d leases=%d", len(ym.aux), len(ym.savedLeases))
		}
		AttentionAlgorithm = "flash"
		yf := ScaledDotProductAttention(q, q, q, nil)
		defer yf.Close()
		if len(yf.aux) != 0 || len(yf.savedLeases) != 0 {
			t.Fatalf("flash NoGrad retained aux=%d leases=%d", len(yf.aux), len(yf.savedLeases))
		}

		x := mustNew(t, seqData(2*8*16, 0.2, 0.2), []int{2, 8, 16}, tensor.CUDA, true)
		w := mustNew(t, seqData(16, 0.5, 0.3), []int{16}, tensor.CUDA, true)
		b := mustNew(t, seqData(16, 0.1, 0.4), []int{16}, tensor.CUDA, true)
		defer x.Close()
		defer w.Close()
		defer b.Close()
		yln := LayerNorm(x, w, b, 1e-5)
		defer yln.Close()
		if len(yln.aux) != 0 {
			t.Fatalf("layernorm NoGrad retained aux=%d", len(yln.aux))
		}
		ygn := GroupNorm(x, w, b, 4, 1e-5)
		defer ygn.Close()
		if len(ygn.aux) != 0 {
			t.Fatalf("groupnorm NoGrad retained aux=%d", len(ygn.aux))
		}
		kw := mustNew(t, seqData(16*16*3, 0.2, 0.5), []int{16, 16, 3}, tensor.CUDA, true)
		defer kw.Close()
		yc := Conv1d(x, kw, b, 1)
		defer yc.Close()
		if len(yc.aux) != 0 || len(yc.savedLeases) != 0 {
			t.Fatalf("conv1d NoGrad retained aux=%d leases=%d", len(yc.aux), len(yc.savedLeases))
		}
	})
}

func TestMultiheadAttentionModesAndMaskExpansion(t *testing.T) {
	devices := []tensor.Device{tensor.CPU}
	if cuda.Available() {
		devices = append(devices, tensor.CUDA)
	}
	for _, dev := range devices {
		t.Run(string(dev), func(t *testing.T) {
			if dev == tensor.CUDA {
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Skip(err)
				}
				defer ctx.Close()
			}
			mha, err := NewMultiheadAttention(16, 4, dev)
			if err != nil {
				t.Fatal(err)
			}
			mha.Training = false
			defer func() {
				for _, p := range mha.Parameters {
					p.Value.Close()
				}
			}()

			xd := seqData(2*5*16, 0.3, 0.2)
			ud := seqData(2*5*16, 0.2, 0.7)
			x1 := mustNew(t, xd, []int{2, 5, 16}, dev, true)
			u1 := mustNew(t, ud, []int{2, 5, 16}, dev, false)
			ySelf := mha.Forward(x1, x1, x1, nil)
			outSelf := mustHost(t, ySelf)
			l1 := weightedSum(ySelf, u1)
			if err := l1.Backward(); err != nil {
				t.Fatal(err)
			}
			gxSelf := mustGradHost(t, x1)
			gpSelf := make([][]float32, len(mha.Parameters))
			for i, p := range mha.Parameters {
				gpSelf[i] = mustGradHost(t, p.Value)
			}
			l1.ReleaseGraph()
			x1.Close()
			u1.Close()
			for _, p := range mha.Parameters {
				p.Value.ZeroGrad()
			}

			q := mustNew(t, xd, []int{2, 5, 16}, dev, true)
			k := mustNew(t, xd, []int{2, 5, 16}, dev, true)
			v := mustNew(t, xd, []int{2, 5, 16}, dev, true)
			u2 := mustNew(t, ud, []int{2, 5, 16}, dev, false)
			ySep := mha.Forward(q, k, v, nil)
			outSep := mustHost(t, ySep)
			l2 := weightedSum(ySep, u2)
			if err := l2.Backward(); err != nil {
				t.Fatal(err)
			}
			gq, gk, gv := mustGradHost(t, q), mustGradHost(t, k), mustGradHost(t, v)
			gxSep := make([]float32, len(gq))
			for i := range gxSep {
				gxSep[i] = gq[i] + gk[i] + gv[i]
			}
			assertCloseSlice(t, "self vs separate out", outSelf, outSep, 1e-4)
			assertCloseSlice(t, "self vs separate gx", gxSelf, gxSep, 1e-4)
			for i, p := range mha.Parameters {
				assertCloseSlice(t, "param "+p.Name, gpSelf[i], mustGradHost(t, p.Value), 1e-4)
			}
			l2.ReleaseGraph()
			q.Close()
			k.Close()
			v.Close()
			u2.Close()

			// Cross-attention with unequal sequence lengths.
			qc := mustNew(t, seqData(2*4*16, 0.3, 0.3), []int{2, 4, 16}, dev, true)
			kc := mustNew(t, seqData(2*7*16, 0.3, 0.6), []int{2, 7, 16}, dev, true)
			vc := mustNew(t, seqData(2*7*16, 0.3, 0.9), []int{2, 7, 16}, dev, true)
			yc := mha.Forward(qc, kc, vc, nil)
			if len(yc.Shape) != 3 || yc.Shape[0] != 2 || yc.Shape[1] != 4 || yc.Shape[2] != 16 {
				t.Fatalf("cross-attention shape %v", yc.Shape)
			}
			lc := Sum(yc, 0, 1, 2)
			if err := lc.Backward(); err != nil {
				t.Fatal(err)
			}
			_ = mustGradHost(t, qc)
			_ = mustGradHost(t, kc)
			_ = mustGradHost(t, vc)
			lc.ReleaseGraph()
			qc.Close()
			kc.Close()
			vc.Close()

			// Mask expanding singleton batch axis must reject size mismatch instead of truncating.
			xs := mustNew(t, seqData(1*3*16, 0.2, 0.1), []int{1, 3, 16}, dev, false)
			badMask := mustNew(t, make([]float32, 2*1*3*3), []int{2, 1, 3, 3}, dev, false)
			defer xs.Close()
			defer badMask.Close()
			panicked := false
			func() {
				defer func() {
					if recover() != nil {
						panicked = true
					}
				}()
				_ = mha.Forward(xs, xs, xs, badMask)
			}()
			if !panicked {
				t.Fatal("expected panic when mask expands singleton batch axis")
			}
		})
	}

	for _, dev := range []tensor.Device{tensor.Vulkan, tensor.DirectML} {
		if !gpu.Available(string(dev)) {
			continue
		}
		t.Run("inference_self_attn_"+string(dev), func(t *testing.T) {
			cpuMHA, err := NewMultiheadAttention(16, 4, tensor.CPU)
			if err != nil {
				t.Fatal(err)
			}
			cpuMHA.Training = false
			devMHA, err := NewMultiheadAttention(16, 4, dev)
			if err != nil {
				t.Fatal(err)
			}
			devMHA.Training = false
			defer func() {
				for _, p := range cpuMHA.Parameters {
					p.Value.Close()
				}
				for _, p := range devMHA.Parameters {
					p.Value.Close()
				}
			}()
			state := cpuMHA.StateDict()
			if err := devMHA.LoadStateDict(state); err != nil {
				t.Fatal(err)
			}
			xd := seqData(2*5*16, 0.3, 0.4)
			xc := mustNew(t, xd, []int{2, 5, 16}, tensor.CPU, false)
			defer xc.Close()
			yc := cpuMHA.Forward(xc, xc, xc, nil)
			defer yc.ReleaseGraph()

			exec := NewExecutionContext()
			exec.BindModule(&devMHA.Module)
			xdv, err := exec.New(xd, []int{2, 5, 16}, dev, false)
			if err != nil {
				t.Fatal(err)
			}
			defer xdv.Close()
			var ydv *Tensor
			exec.NoGrad(func() {
				ydv = devMHA.Forward(xdv, xdv, xdv, nil)
			})
			defer ydv.ReleaseGraph()
			assertCloseSlice(t, string(dev)+" self-attn", mustHost(t, ydv), yc.Data, 1e-4)
		})
	}
}

func TestCPUNormParallelLarge(t *testing.T) {
	rows, cols := 512, 256
	xd := seqData(rows*cols, 0.6, 0.2)
	wd := seqData(cols, 0.4, 0.5)
	bd := seqData(cols, 0.1, 0.9)
	ud := seqData(rows*cols, 0.3, 1.3)
	x := mustNew(t, xd, []int{16, 32, cols}, tensor.CPU, true)
	w := mustNew(t, wd, []int{cols}, tensor.CPU, true)
	b := mustNew(t, bd, []int{cols}, tensor.CPU, true)
	u := mustNew(t, ud, []int{16, 32, cols}, tensor.CPU, false)
	defer x.Close()
	defer w.Close()
	defer b.Close()
	defer u.Close()

	y := LayerNorm(x, w, b, 1e-5)
	loss := weightedSum(y, u)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	wantY := make([]float32, len(xd))
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			sum += float64(xd[r*cols+c])
		}
		mean := sum / float64(cols)
		var ss float64
		for c := 0; c < cols; c++ {
			d := float64(xd[r*cols+c]) - mean
			ss += d * d
		}
		inv := 1 / math.Sqrt(ss/float64(cols)+1e-5)
		for c := 0; c < cols; c++ {
			wantY[r*cols+c] = float32((float64(xd[r*cols+c])-mean)*inv*float64(wd[c]) + float64(bd[c]))
		}
	}
	assertCloseSlice(t, "large cpu layernorm", y.Data, wantY, 1e-4)
	loss.ReleaseGraph()

	x.ZeroGrad()
	w.ZeroGrad()
	b.ZeroGrad()
	yg := GroupNorm(x, w, b, 8, 1e-5)
	lg := weightedSum(yg, u)
	if err := lg.Backward(); err != nil {
		t.Fatal(err)
	}
	wantG := make([]float32, len(xd))
	bs, tm, ch, groups := 16, 32, cols, 8
	cg := ch / groups
	cnt := float64(tm * cg)
	for n := 0; n < bs; n++ {
		for g := 0; g < groups; g++ {
			var sum float64
			for at := 0; at < tm; at++ {
				for c := g * cg; c < (g+1)*cg; c++ {
					sum += float64(xd[(n*tm+at)*ch+c])
				}
			}
			mean := sum / cnt
			var ss float64
			for at := 0; at < tm; at++ {
				for c := g * cg; c < (g+1)*cg; c++ {
					d := float64(xd[(n*tm+at)*ch+c]) - mean
					ss += d * d
				}
			}
			inv := 1 / math.Sqrt(ss/cnt+1e-5)
			for at := 0; at < tm; at++ {
				for c := g * cg; c < (g+1)*cg; c++ {
					idx := (n*tm+at)*ch + c
					wantG[idx] = float32((float64(xd[idx])-mean)*inv*float64(wd[c]) + float64(bd[c]))
				}
			}
		}
	}
	assertCloseSlice(t, "large cpu groupnorm", yg.Data, wantG, 1e-4)
	lg.ReleaseGraph()
}

func TestCUDAFastPathParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Skip(err)
	}
	defer ctx.Close()

	t.Run("reduce_last_axis", func(t *testing.T) {
		shapes := [][]int{{5, 33}, {3, 4, 100}, {2, 3, 4, 65}}
		for _, shape := range shapes {
			axis := len(shape) - 1
			xd := seqData(numel(shape), 0.5, 0.2)
			outShape := shape[:axis]
			ud := seqData(numel(outShape), 0.3, 0.7)
			for _, mean := range []bool{false, true} {
				run := func(dev tensor.Device) ([]float32, []float32) {
					x := mustNew(t, xd, shape, dev, true)
					u := mustNew(t, ud, outShape, dev, false)
					var y *Tensor
					if mean {
						y = Mean(x, axis)
					} else {
						y = Sum(x, axis)
					}
					out := mustHost(t, y)
					loss := weightedSum(y, u)
					if err := loss.Backward(); err != nil {
						t.Fatal(err)
					}
					gx := mustGradHost(t, x)
					loss.ReleaseGraph()
					x.Close()
					u.Close()
					return out, gx
				}
				wy, wg := run(tensor.CPU)
				gy, gg := run(tensor.CUDA)
				assertCloseSlice(t, "out", gy, wy, 1e-4)
				assertCloseSlice(t, "grad", gg, wg, 1e-4)
			}
		}
		big := []float32{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}
		xb := mustNew(t, big, []int{2, 2}, tensor.CUDA, false)
		yb := Mean(xb, 1)
		gotBig := mustHost(t, yb)
		yb.ReleaseGraph()
		xb.Close()
		for i, v := range gotBig {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Fatalf("MaxFloat32 mean[%d] = %g, want finite", i, v)
			}
		}
	})

	t.Run("binary_same_and_alias", func(t *testing.T) {
		xd := seqData(257, 0.5, 0.4)
		yd := seqData(257, 0.5, 1.1)
		for i := range yd {
			if math.Abs(float64(yd[i])) < 0.2 {
				yd[i] = 0.5
			}
		}
		ud := seqData(257, 0.3, 2.0)
		ops := []struct {
			name string
			fn   func(*Tensor, *Tensor) *Tensor
		}{
			{"add", Add},
			{"sub", Sub},
			{"mul", Mul},
			{"div", Div},
			{"max", Max},
			{"min", Min},
		}
		for _, op := range ops {
			run := func(dev tensor.Device, alias bool) ([]float32, []float32, []float32) {
				a := mustNew(t, xd, []int{257}, dev, true)
				var b *Tensor
				if alias {
					b = a
				} else {
					b = mustNew(t, yd, []int{257}, dev, true)
				}
				u := mustNew(t, ud, []int{257}, dev, false)
				y := op.fn(a, b)
				out := mustHost(t, y)
				loss := weightedSum(y, u)
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				ga := mustGradHost(t, a)
				var gb []float32
				if !alias {
					gb = mustGradHost(t, b)
				}
				loss.ReleaseGraph()
				u.Close()
				if !alias {
					b.Close()
				}
				a.Close()
				return out, ga, gb
			}
			for _, alias := range []bool{false, true} {
				if alias && (op.name == "div" || op.name == "max" || op.name == "min") {
					continue
				}
				wo, wga, wgb := run(tensor.CPU, alias)
				go_, gga, ggb := run(tensor.CUDA, alias)
				assertCloseSlice(t, op.name+"_out", go_, wo, 1e-4)
				assertCloseSlice(t, op.name+"_ga", gga, wga, 1e-4)
				if !alias {
					assertCloseSlice(t, op.name+"_gb", ggb, wgb, 1e-4)
				}
			}
		}
	})

	t.Run("bias_add_multi_block", func(t *testing.T) {
		for _, tc := range []struct {
			aShape []int
			bShape []int
		}{
			{[]int{16, 37}, []int{37}},
			{[]int{8, 64, 100}, []int{100}},
			{[]int{4, 32, 65}, []int{1, 65}},
			{[]int{0, 16}, []int{16}},
		} {
			ad := seqData(numel(tc.aShape), 0.4, 0.2)
			bd := seqData(numel(tc.bShape), 0.4, 0.8)
			ud := seqData(numel(tc.aShape), 0.3, 1.5)
			run := func(dev tensor.Device) ([]float32, []float32, []float32) {
				a := mustNew(t, ad, tc.aShape, dev, true)
				b := mustNew(t, bd, tc.bShape, dev, true)
				u := mustNew(t, ud, tc.aShape, dev, false)
				y := Add(a, b)
				out := mustHost(t, y)
				loss := weightedSum(y, u)
				if err := loss.Backward(); err != nil {
					t.Fatal(err)
				}
				ga := mustGradHost(t, a)
				gb := mustGradHost(t, b)
				loss.ReleaseGraph()
				a.Close()
				b.Close()
				u.Close()
				return out, ga, gb
			}
			wo, wga, wgb := run(tensor.CPU)
			go_, gga, ggb := run(tensor.CUDA)
			assertCloseSlice(t, "out", go_, wo, 1e-4)
			assertCloseSlice(t, "ga", gga, wga, 1e-4)
			assertCloseSlice(t, "gb", ggb, wgb, 5e-3)
		}
	})

	t.Run("cross_entropy_warp", func(t *testing.T) {
		rows, classes := 19, 137
		xd := seqData(rows*classes, 1.5, 0.3)
		targets := make([]int, rows)
		for i := range targets {
			if i%5 == 0 {
				targets[i] = -100
			} else {
				targets[i] = (i * 17) % classes
			}
		}
		run := func(dev tensor.Device) ([]float32, []float32) {
			x := mustNew(t, xd, []int{rows, classes}, dev, true)
			loss := CrossEntropy(x, targets, -100)
			out := mustHost(t, loss)
			if err := loss.Backward(); err != nil {
				t.Fatal(err)
			}
			gx := mustGradHost(t, x)
			loss.ReleaseGraph()
			x.Close()
			return out, gx
		}
		wo, wg := run(tensor.CPU)
		go_, gg := run(tensor.CUDA)
		assertCloseSlice(t, "ce_out", go_, wo, 1e-4)
		assertCloseSlice(t, "ce_grad", gg, wg, 1e-4)
	})

	t.Run("matmul_transposed_b_and_direct", func(t *testing.T) {
		defer func() { BF16Autocast = false }()
		for _, bf16 := range []bool{false, true} {
			BF16Autocast = bf16
			tol := 1e-4
			if bf16 {
				tol = 3e-2
			}
			for _, direct := range []bool{false, true} {
				for _, aShape := range [][]int{{7, 17}, {3, 5, 17}} {
					ad := seqData(numel(aShape), 0.3, 0.1)
					wd := seqData(13*17, 0.3, 0.6)
					oShape := append(append([]int(nil), aShape[:len(aShape)-1]...), 13)
					ud := seqData(numel(oShape), 0.3, 1.2)
					run := func(dev tensor.Device) ([]float32, []float32, []float32) {
						a := mustNew(t, ad, aShape, dev, true)
						w := mustNew(t, wd, []int{13, 17}, dev, true)
						u := mustNew(t, ud, oShape, dev, false)
						var y *Tensor
						if direct {
							y = matMulTransB(a, w)
						} else {
							y = MatMul(a, Transpose(w, 0, 1))
						}
						out := mustHost(t, y)
						loss := weightedSum(y, u)
						if err := loss.Backward(); err != nil {
							t.Fatal(err)
						}
						ga := mustGradHost(t, a)
						gw := mustGradHost(t, w)
						loss.ReleaseGraph()
						a.Close()
						w.Close()
						u.Close()
						return out, ga, gw
					}
					wo, wga, wgw := run(tensor.CPU)
					go_, gga, ggw := run(tensor.CUDA)
					assertCloseSlice(t, "out", go_, wo, tol)
					assertCloseSlice(t, "ga", gga, wga, tol)
					assertCloseSlice(t, "gw", ggw, wgw, tol)
				}
			}
		}
	})
}
