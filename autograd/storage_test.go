package autograd

import (
	"strings"
	"sync"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestStorageSharedConstantCPUWorkers(t *testing.T) {
	constant := Must([]float32{2, 3}, []int{2}, false)
	defer constant.Close()
	const workers = 8
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			context := NewExecutionContext()
			x, err := context.New([]float32{1, 4}, []int{2}, tensor.CPU, true)
			if err != nil {
				t.Error(err)
				return
			}
			defer x.Close()
			for i := 0; i < 30; i++ {
				x.ZeroGrad()
				loss := Sum(Mul(x, constant), 0)
				if err := loss.Backward(); err != nil {
					t.Error(err)
					loss.ReleaseGraph()
					return
				}
				loss.ReleaseGraph()
				if len(x.Grad) != 2 || x.Grad[0] != 2 || x.Grad[1] != 3 {
					t.Error("shared-constant gradient changed")
					return
				}
			}
		}()
	}
	wg.Wait()
	if constant.children.Load() != 0 || constant.closed.Load() {
		t.Fatal("shared read-only constant lifetime")
	}
}

func TestStorageInferenceReturnsAllTemporaries(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{.1, .2, .3, .4, .5, .6, .7, .8}, []int{1, 4, 2}, device, true)
		w := conv2dTestTensor(t, []float32{.1, .2, .3, .4, .5, .6, .6, .5, .4, .3, .2, .1}, []int{2, 2, 3}, device, true)
		b := conv2dTestTensor(t, []float32{.1, .2}, []int{2}, device, true)
		before := cuda.MemoryStats().LiveBytes
		for i := 0; i < 20; i++ {
			NoGrad(func() {
				y := LayerNorm(Conv1dGEMM(x, w, b, 1), b, b, .01)
				view := Transpose(Reshape(y, 4, 2), 0, 1)
				if _, err := view.ToHost(); err != nil {
					t.Fatal(err)
				}
				view.ReleaseGraph()
				view.Close()
				y.Close()
			})
		}
		if x.children.Load() != 0 || w.children.Load() != 0 || b.children.Load() != 0 {
			t.Fatal("inference edges leaked")
		}
		if device == tensor.CUDA && cuda.MemoryStats().LiveBytes != before {
			t.Fatal("inference live device bytes grew")
		}
	})
}

func storageDevices(t *testing.T, fn func(*testing.T, tensor.Device)) {
	t.Helper()
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Fatal(err)
				}
				defer ctx.Close()
			}
			fn(t, device)
		})
	}
}

func TestStorageViewSurvivesBaseClose(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device, true)
		base := AddScalar(x, 1)
		view := Transpose(base, 0, 1)
		allocation := base.storage
		base.Close()
		base.Close()
		if _, err := base.ToHost(); err == nil {
			t.Fatal("closed base remains publicly usable")
		}
		if base.disposed.Load() {
			t.Fatal("dependent view did not protect base")
		}
		// A same-size allocation exercises actual pool reuse. It must not
		// overwrite the view's borrowed data after closing the base handle.
		noise := conv2dTestTensor(t, []float32{99, 99, 99, 99, 99, 99}, []int{2, 3}, device, false)
		defer noise.Close()
		conv2dClose(t, "surviving view", conv2dHost(t, view, false), []float32{2, 5, 3, 6, 4, 7}, 0, 0)
		loss := Sum(Mul(view, view), 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		loss.ReleaseGraph()
		conv2dClose(t, "closed ancestor gradient", conv2dHost(t, x, true), []float32{4, 6, 8, 10, 12, 14}, 0, 0)
		view.Close()
		view.Close()
		base.ReleaseGraph()
		if allocation.references.Load() != 0 || !base.disposed.Load() {
			t.Fatal("last view did not release allocation")
		}
		if x.children.Load() != 0 {
			t.Fatal("parent edge leaked")
		}
	})
}

func TestStorageBranchReleaseIsolation(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{2, 3}, []int{2}, device, true)
		shared := Mul(x, x)
		allocation := shared.storage
		left := Sum(shared, 0)
		right := Sum(MulScalar(shared, 3), 0)
		left.ReleaseGraph()
		left.ReleaseGraph()
		if shared.disposed.Load() || shared.historyReleased {
			t.Fatal("releasing one branch destroyed another")
		}
		conv2dClose(t, "shared values", conv2dHost(t, shared, false), []float32{4, 9}, 0, 0)
		if err := right.Backward(); err != nil {
			t.Fatal(err)
		}
		right.ReleaseGraph()
		shared.Close()
		conv2dClose(t, "remaining branch gradient", conv2dHost(t, x, true), []float32{12, 18}, 0, 0)
		if allocation.references.Load() != 0 || x.children.Load() != 0 {
			t.Fatal("shared graph leaked")
		}
	})
}

func TestStorageRetainedViewData(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, device, true)
		base := Mul(x, x)
		allocation := base.storage
		view := Transpose(base, 0, 1)
		view.RetainData()
		loss := Sum(view, 0, 1)
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		loss.ReleaseGraph()
		if !base.disposed.Load() {
			t.Fatal("retained view unnecessarily retained base graph")
		}
		if allocation.references.Load() != 1 || len(view.parents) != 0 {
			t.Fatal("only retained value lease should remain")
		}
		conv2dClose(t, "retained view", conv2dHost(t, view, false), []float32{1, 9, 4, 16}, 0, 0)
		view.ReleaseGraph()
		conv2dClose(t, "repeat release", conv2dHost(t, view, false), []float32{1, 9, 4, 16}, 0, 0)
		invalid := Sum(view, 0, 1)
		if err := invalid.Backward(); err == nil {
			t.Fatal("value retention accidentally retained history")
		}
		invalid.ReleaseGraph()
		// Detach converts the retained value to an independent leaf for reuse.
		detached := view.Detach()
		defer detached.Close()
		view.Close()
		view.Close()
		conv2dClose(t, "independent detach", conv2dHost(t, detached, false), []float32{1, 9, 4, 16}, 0, 0)
		if allocation.references.Load() != 0 {
			t.Fatal("retained view allocation leaked")
		}
	})
}

func TestStorageRetainedBackwardBranches(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{2, 3}, []int{2}, device, true)
		shared := Mul(x, x)
		shared.RetainGrad()
		defer shared.Close()
		left := Sum(shared, 0)
		right := Sum(MulScalar(shared, 3), 0)
		defer left.ReleaseGraph()
		defer right.ReleaseGraph()
		if err := left.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "first leaf gradient", conv2dHost(t, x, true), []float32{4, 6}, 0, 0)
		conv2dClose(t, "first intermediate gradient", conv2dHost(t, shared, true), []float32{1, 1}, 0, 0)
		if err := right.Backward(); err != nil {
			t.Fatal(err)
		}
		conv2dClose(t, "accumulated leaf gradient", conv2dHost(t, x, true), []float32{16, 24}, 0, 0)
		conv2dClose(t, "accumulated retained gradient", conv2dHost(t, shared, true), []float32{4, 4}, 0, 0)
		if err := left.Backward(); err == nil || !strings.Contains(err.Error(), "history") {
			t.Fatalf("reusing consumed history: %v", err)
		}
		conv2dClose(t, "failed backward was atomic", conv2dHost(t, x, true), []float32{16, 24}, 0, 0)
	})
}

// Saved im2col, GroupNorm statistics and masked-loss accumulators all need to
// survive the first retained pass, on both devices. The closed target has a
// lifetime-only edge: no target gradient is computed.
func TestStorageRetainsBackwardAuxiliaries(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{.1, .2, .3, .4, .5, .6, .7, .8}, []int{1, 4, 2}, device, true)
		w := conv2dTestTensor(t, []float32{.1, .2, .3, .4, .5, .6, .6, .5, .4, .3, .2, .1}, []int{2, 2, 3}, device, true)
		b := conv2dTestTensor(t, []float32{.1, .2}, []int{2}, device, true)
		gamma := conv2dTestTensor(t, []float32{.9, 1.1}, []int{2}, device, true)
		beta := conv2dTestTensor(t, []float32{.1, .2}, []int{2}, device, true)
		target := conv2dTestTensor(t, []float32{-.2, .4, .1, -.3, .5, .2, -.4, .7}, []int{1, 4, 2}, device, false)
		prediction := GroupNorm(Conv1dGEMM(x, w, b, 1), gamma, beta, 1, .01)
		prediction = LayerNorm(prediction, gamma, beta, .03)
		loss := MaskedLoss(prediction, target, true)
		defer loss.ReleaseGraph()
		target.Close()
		if target.disposed.Load() {
			t.Fatal("closed target was not saved")
		}
		params := []*Tensor{x, w, b, gamma, beta}
		if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		first := make([][]float32, len(params))
		for i, p := range params {
			first[i] = conv2dHost(t, p, true)
		}
		// Reuse every relevant scratch size between passes. A saved statistic
		// accidentally returned to a pool will be overwritten, not just left
		// untouched until the second backward happens to read it again.
		if device == tensor.CPU {
			var scratch [][]float32
			for i := 0; i < 16; i++ {
				for _, n := range []int{1, 4, 8, 24} {
					data := cpuAlloc(n)
					for j := range data {
						data[j] = 97
					}
					scratch = append(scratch, data)
				}
			}
			for _, data := range scratch {
				cpuRelease(data)
			}
		} else {
			for i := 0; i < 16; i++ {
				buf := mustAlloc(24)
				if err := buf.Memset(0, buf.Size()); err != nil {
					t.Fatal(err)
				}
				buf.Free()
			}
		}
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		for i, p := range params {
			for j := range first[i] {
				first[i][j] *= 2
			}
			conv2dClose(t, "repeated saved-state gradient", conv2dHost(t, p, true), first[i], 2e-5, 2e-5)
		}
		loss.ReleaseGraph()
		if !target.disposed.Load() {
			t.Fatal("saved target leaked")
		}
	})
}

func TestStorageMutationInvalidatesSavedGraph(t *testing.T) {
	storageDevices(t, func(t *testing.T, device tensor.Device) {
		x := conv2dTestTensor(t, []float32{2, 3}, []int{2}, device, true)
		loss := Sum(Mul(x, x), 0)
		defer loss.ReleaseGraph()
		alias := Reshape(x, 2)
		defer alias.Close()
		if err := alias.CopyFrom([]float32{4, 5}); err != nil {
			t.Fatal(err)
		}
		if err := loss.Backward(); err == nil || !strings.Contains(err.Error(), "modified") {
			t.Fatalf("mutation not detected: %v", err)
		}
		if len(conv2dHost(t, x, true)) != 0 {
			t.Fatal("invalid backward partially changed gradients")
		}
		// Optimizer writes are versioned too, so a retained graph cannot use
		// pre-update values silently on its second pass.
		p := conv2dTestTensor(t, []float32{2, 3}, []int{2}, device, true)
		retained := Sum(Mul(p, p), 0)
		defer retained.ReleaseGraph()
		if err := retained.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
			t.Fatal(err)
		}
		opt := NewAdamW([]Parameter{{"p", p}}, .01, 0)
		defer opt.Close()
		opt.Step()
		if err := retained.Backward(); err == nil || !strings.Contains(err.Error(), "modified") {
			t.Fatalf("optimizer mutation not detected: %v", err)
		}
	})
}

func TestStorageBF16SharedView(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	old := BF16Autocast
	BF16Autocast = true
	defer func() { BF16Autocast = old }()
	x := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, tensor.CUDA, false)
	base := Add(x, x)
	view := Reshape(base, 2, 2)
	view.RetainData()
	shadow := view.currentBF16()
	if shadow == nil {
		t.Fatal("producer did not generate BF16")
	}
	base.ReleaseGraph()
	view.ReleaseGraph()
	if !base.disposed.Load() || shadow.Pointer() == 0 {
		t.Fatal("shared BF16 shadow was freed")
	}
	identity := conv2dTestTensor(t, []float32{1, 0, 0, 1}, []int{2, 2}, tensor.CUDA, false)
	out := MatMul(view, identity)
	conv2dClose(t, "surviving BF16", conv2dHost(t, out, false), []float32{2, 4, 6, 8}, 0, 0)
	out.ReleaseGraph()
	view.Close()
	if shadow.Pointer() != 0 {
		t.Fatal("BF16 allocation leaked")
	}
	// A view mutation invalidates the cache on every alias of its storage.
	parent := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, tensor.CUDA, false)
	parent.ensureBF16()
	alias := Reshape(parent, 4)
	if err := alias.CopyFrom([]float32{2, 3, 4, 5}); err != nil {
		t.Fatal(err)
	}
	if parent.currentBF16() != nil {
		t.Fatal("parent BF16 cache survived alias mutation")
	}
	updated := MatMul(parent, identity)
	conv2dClose(t, "updated BF16", conv2dHost(t, updated, false), []float32{2, 3, 4, 5}, 0, 0)
	updated.ReleaseGraph()
	alias.Close()
}

func TestStorageIndexBufferSavedLease(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	w := conv2dTestTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, tensor.CUDA, true)
	ids, err := NewIndexBuffer([]int{1, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	buffer := ids.buffer
	loss := Sum(EmbeddingFromIndexBuffer(w, ids, []int{3}, -1), 0, 1)
	defer loss.ReleaseGraph()
	ids.Close()
	ids.Close()
	if buffer.Pointer() == 0 {
		t.Fatal("indices freed before backward")
	}
	if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: true}); err != nil {
		t.Fatal(err)
	}
	if buffer.Pointer() == 0 {
		t.Fatal("indices freed during retained backward")
	}
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "saved-index gradients", conv2dHost(t, w, true), []float32{0, 0, 4, 4, 2, 2}, 0, 0)
	if buffer.Pointer() != 0 {
		t.Fatal("saved indices leaked")
	}
}
