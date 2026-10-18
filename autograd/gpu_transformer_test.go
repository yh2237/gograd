package autograd

import (
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	"github.com/yh2237/gograd/vulkan"
)

// CPU calculations below are exclusively the driver-free test fixture.
func (b *inferenceTestBackend) reference(out *gpu.Buffer, shape []int, inputs []*gpu.Buffer, shapes [][]int, run func([]*Tensor) *Tensor) error {
	b.mu.Lock()
	fail := b.failNext
	b.failNext = false
	b.mu.Unlock()
	if fail {
		return injectedInferenceError
	}
	values := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		x, err := b.memory(input, input.Len())
		if err != nil {
			return err
		}
		v, err := New(x.data, shapes[i], tensor.CPU, false)
		if err != nil {
			return err
		}
		defer v.Close()
		values[i] = v
	}
	r := run(values)
	defer r.ReleaseGraph()
	y, err := b.memory(out, numel(shape))
	if err != nil {
		return err
	}
	host, err := r.ToHost()
	if err != nil {
		return err
	}
	copy(y.data, host)
	return nil
}
func (b *inferenceTestBackend) GELU(out, a *gpu.Buffer, n int) error {
	return b.reference(out, []int{n}, []*gpu.Buffer{a}, [][]int{{n}}, func(x []*Tensor) *Tensor { return GELU(x[0], true) })
}
func (b *inferenceTestBackend) GELUExact(out, a *gpu.Buffer, n int) error {
	return b.reference(out, []int{n}, []*gpu.Buffer{a}, [][]int{{n}}, func(x []*Tensor) *Tensor { return GELU(x[0], false) })
}
func (b *inferenceTestBackend) Softmax(out, a *gpu.Buffer, rows, cols int) error {
	return b.reference(out, []int{rows, cols}, []*gpu.Buffer{a}, [][]int{{rows, cols}}, func(x []*Tensor) *Tensor { return Softmax(x[0], -1) })
}
func (b *inferenceTestBackend) LayerNorm(out, a, w, bias *gpu.Buffer, rows, cols int, eps float32) error {
	return b.reference(out, []int{rows, cols}, []*gpu.Buffer{a, w, bias}, [][]int{{rows, cols}, {cols}, {cols}}, func(x []*Tensor) *Tensor { return LayerNorm(x[0], x[1], x[2], eps) })
}
func (b *inferenceTestBackend) GroupNorm(out, a, w, bias *gpu.Buffer, batch, frames, channels, groups int, eps float32) error {
	return b.reference(out, []int{batch, frames, channels}, []*gpu.Buffer{a, w, bias}, [][]int{{batch, frames, channels}, {channels}, {channels}}, func(x []*Tensor) *Tensor { return GroupNorm(x[0], x[1], x[2], groups, eps) })
}
func (b *inferenceTestBackend) Conv1d(out, a, w, bias *gpu.Buffer, o gpu.Conv1dOptions) error {
	length, err := o.OutputLength()
	if err != nil {
		return err
	}
	inputs, shapes := []*gpu.Buffer{a, w}, [][]int{{o.Batch, o.InputLength, o.InChannels}, {o.OutChannels, o.InChannels, o.Kernel}}
	if bias != nil {
		inputs = append(inputs, bias)
		shapes = append(shapes, []int{o.OutChannels})
	}
	return b.reference(out, []int{o.Batch, length, o.OutChannels}, inputs, shapes, func(x []*Tensor) *Tensor {
		var v *Tensor
		if bias != nil {
			v = x[2]
		} else {
			v, _ = Zeros([]int{o.OutChannels}, tensor.CPU, false)
			defer v.Close()
		}
		return Conv1dGEMMStride(x[0], x[1], v, o.Stride, o.Padding, o.Dilation)
	})
}

// leakCheck is registered first so it runs after graphs and persistent leaves.
func inferenceLeakCheck(t *testing.T, device tensor.Device) {
	t.Helper()
	live, _, _ := inferenceFixture.snapshot()
	stats := vulkan.MemoryStats()
	t.Cleanup(func() {
		if device == fixtureDevice {
			if after, _, _ := inferenceFixture.snapshot(); after != live {
				t.Errorf("fixture leaked: %d -> %d", live, after)
			}
		} else if device == tensor.Vulkan {
			if after := vulkan.MemoryStats(); after.LiveBytes != stats.LiveBytes || after.LiveBuffers != stats.LiveBuffers {
				t.Errorf("Vulkan leaked: %+v -> %+v", stats, after)
			}
		}
	})
}

func inferenceTestNew(t *testing.T, device tensor.Device, shape []int, seed int) *Tensor {
	t.Helper()
	data := make([]float32, numel(shape))
	for i := range data {
		data[i] = float32((i*13+seed*17)%97-48) / 32
	}
	x, err := New(data, shape, device, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(x.Close)
	return x
}

func TestInferenceGPUTransformerOps(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		inferenceLeakCheck(t, device)
		for _, op := range []string{"gelu", "gelu_exact", "bias_gelu", "bias_residual", "softmax_last", "softmax_middle", "softmax_first", "layernorm", "groupnorm", "concat", "conv_same", "conv_stride", "conv_no_bias"} {
			t.Run(op, func(t *testing.T) {
				inferenceLeakCheck(t, device)
				build := func(d tensor.Device) *Tensor {
					x := inferenceTestNew(t, d, []int{2, 8, 7}, 1)
					x = Transpose(x, 1, 2)
					w, bias := inferenceTestNew(t, d, []int{8}, 2), inferenceTestNew(t, d, []int{8}, 3)
					switch op {
					case "gelu":
						return GELU(x, true)
					case "gelu_exact":
						return GELU(x, false)
					case "bias_gelu":
						return BiasGELU(x, bias)
					case "bias_residual":
						return BiasResidual(x, bias, x)
					case "softmax_last":
						return Softmax(x, -1)
					case "softmax_middle":
						return Softmax(x, 1)
					case "softmax_first":
						return Softmax(x, 0)
					case "layernorm":
						return LayerNorm(x, w, bias, 1e-5)
					case "groupnorm":
						return GroupNorm(x, w, bias, 4, 1e-5)
					case "concat":
						return Concat(-1, Slice(x, 2, 1, 4), Slice(x, 2, 0, 0), Slice(x, 2, 0, 8), Slice(x, 2, 5, 8))
					default:
						cw, cb := inferenceTestNew(t, d, []int{5, 8, 3}, 4), inferenceTestNew(t, d, []int{5}, 5)
						if op == "conv_same" {
							return Conv1d(x, cw, cb, 2)
						}
						if op == "conv_stride" {
							return Conv1dGEMMStride(x, cw, cb, 2, 2, 2)
						}
						if d == tensor.CPU {
							zero, err := Zeros([]int{5}, d, false)
							if err != nil {
								t.Fatal(err)
							}
							t.Cleanup(zero.Close)
							return Conv1dGEMMStride(x, cw, zero, 2, 2, 2)
						}
						return Conv1dGEMMStride(x, cw, nil, 2, 2, 2)
					}
				}
				_, _, downloads := inferenceFixture.snapshot()
				out := build(device)
				t.Cleanup(out.ReleaseGraph)
				if device == fixtureDevice {
					if _, _, after := inferenceFixture.snapshot(); after != downloads {
						t.Fatal("operation downloaded inputs")
					}
				}
				ref := build(tensor.CPU)
				t.Cleanup(ref.ReleaseGraph)
				host, err := out.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				want, err := ref.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				conv2dClose(t, op, host, want, 1e-4, 1e-4)
			})
		}
	})
}

func TestInferenceGPUTransformerNetwork(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		inferenceLeakCheck(t, device)
		build := func(d tensor.Device) *Tensor {
			x := inferenceTestNew(t, d, []int{2, 17, 4}, 1)
			cw, cb := inferenceTestNew(t, d, []int{8, 4, 3}, 2), inferenceTestNew(t, d, []int{8}, 3)
			x = GELU(Conv1dGEMMStride(x, cw, cb, 2, 1, 1), true)
			gnw, gnb := inferenceTestNew(t, d, []int{8}, 4), inferenceTestNew(t, d, []int{8}, 5)
			x = GroupNorm(x, gnw, gnb, 2, 1e-5)
			project := func(seed int) *Tensor {
				w, bias := inferenceTestNew(t, d, []int{8, 8}, seed), inferenceTestNew(t, d, []int{8}, seed+1)
				return Add(MatMul(x, w), bias)
			}
			q, k, v := project(6), project(8), project(10)
			split := func(a *Tensor) *Tensor { return Permute(Reshape(a, 2, 9, 2, 4), 0, 2, 1, 3) }
			maskData := make([]float32, 9*9)
			for i := 0; i < 9; i++ {
				for j := i + 1; j < 9; j++ {
					maskData[i*9+j] = float32(math.Inf(-1))
				}
			}
			mask, err := New(maskData, []int{9, 9}, d, false)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(mask.Close)
			attn := ScaledDotProductAttention(split(q), split(k), split(v), mask)
			attn = Reshape(Permute(attn, 0, 2, 1, 3), 2, 9, 8)
			lnw, lnb := inferenceTestNew(t, d, []int{8}, 12), inferenceTestNew(t, d, []int{8}, 13)
			h := LayerNorm(Add(x, attn), lnw, lnb, 1e-5)
			w1, b1 := inferenceTestNew(t, d, []int{8, 19}, 14), inferenceTestNew(t, d, []int{19}, 15)
			w2, b2 := inferenceTestNew(t, d, []int{19, 8}, 16), inferenceTestNew(t, d, []int{8}, 17)
			ff := Add(MatMul(GELU(Add(MatMul(h, w1), b1), true), w2), b2)
			y := Dropout(Add(h, ff), .1, 123, false)
			return Concat(1, Slice(y, 1, 0, 4), Slice(y, 1, 4, 9))
		}
		for iteration := 0; iteration < 3; iteration++ {
			_, _, downloads := inferenceFixture.snapshot()
			out := build(device)
			t.Cleanup(out.ReleaseGraph)
			if device == fixtureDevice {
				if _, _, after := inferenceFixture.snapshot(); after != downloads {
					t.Fatal("network downloaded intermediates")
				}
			}
			ref := build(tensor.CPU)
			t.Cleanup(ref.ReleaseGraph)
			host, err := out.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			want, err := ref.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			conv2dClose(t, "transformer", host, want, 1e-4, 1e-4)
			var dot, xx, yy float64
			for i, v := range host {
				dot += float64(v) * float64(want[i])
				xx += float64(v) * float64(v)
				yy += float64(want[i]) * float64(want[i])
			}
			cosine := dot / math.Sqrt(xx*yy)
			if math.IsNaN(cosine) || cosine < .999 {
				t.Fatalf("cosine=%g", cosine)
			}
			if iteration == 0 {
				t.Logf("device=%s cosine=%.12f", device, cosine)
			}
			out.ReleaseGraph()
			ref.ReleaseGraph()
		}
	})
}

func TestInferenceGPUEncoderLayer(t *testing.T) {
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		inferenceLeakCheck(t, device)
		c, cc := NewExecutionContext(), NewExecutionContext()
		layer, err := NewTransformerEncoderLayer(8, 2, 19, device)
		if err != nil {
			t.Fatal(err)
		}
		cpu, err := NewTransformerEncoderLayer(8, 2, 19, tensor.CPU)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range []*TransformerEncoderLayer{layer, cpu} {
			m.Training = false
			m.Attention.Training = false
			for _, p := range m.NamedParameters() {
				t.Cleanup(p.Value.Close)
			}
		}
		if err := c.BindModule(&layer.Module); err != nil {
			t.Fatal(err)
		}
		if err := cc.BindModule(&cpu.Module); err != nil {
			t.Fatal(err)
		}
		x := inferenceTestNew(t, device, []int{2, 5, 8}, 7)
		cx := inferenceTestNew(t, tensor.CPU, []int{2, 5, 8}, 7)
		if err := c.Bind(x); err != nil {
			t.Fatal(err)
		}
		if err := cc.Bind(cx); err != nil {
			t.Fatal(err)
		}
		_, allocations, _ := inferenceFixture.snapshot()
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { layer.Forward(x, nil, 0) })
		c.NoGrad(func() {
			layer.Training = true
			assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { layer.Forward(x, nil, 0) })
			layer.Training = false
			layer.Attention.Training = true
			assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { layer.Forward(x, nil, 0) })
			assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { layer.Attention.Forward(x, x, x, nil) })
			layer.Attention.Training = false
		})
		if device == fixtureDevice {
			if _, after, _ := inferenceFixture.snapshot(); after != allocations {
				t.Fatal("encoder training guard allocated buffers")
			}
		}
		live, _, downloads := inferenceFixture.snapshot()
		stats := vulkan.MemoryStats()
		for i := 0; i < 3; i++ {
			var y, ref *Tensor
			c.NoGrad(func() { y = layer.Forward(x, nil, 0) })
			t.Cleanup(y.ReleaseGraph)
			cc.NoGrad(func() { ref = cpu.Forward(cx, nil, 0) })
			t.Cleanup(ref.ReleaseGraph)
			if device == fixtureDevice {
				if _, _, reads := inferenceFixture.snapshot(); reads != downloads {
					t.Fatal("encoder downloaded intermediates")
				}
			}
			host, err := y.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			conv2dClose(t, "encoder layer", host, ref.Data, 1e-4, 1e-4)
			y.ReleaseGraph()
			ref.ReleaseGraph()
			if device == fixtureDevice {
				after, _, reads := inferenceFixture.snapshot()
				if after != live {
					t.Fatal("encoder leaked intermediates")
				}
				downloads = reads
			} else {
				after := vulkan.MemoryStats()
				if after.LiveBuffers != stats.LiveBuffers || after.LiveBytes != stats.LiveBytes {
					t.Fatal("encoder leaked GPU intermediates")
				}
			}
		}
	})
}

func TestInferenceGPUTransformerGuards(t *testing.T) {
	c := NewExecutionContext()
	x, err := c.New([]float32{1, 2, 3, 4}, []int{1, 2, 2}, fixtureDevice, true)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	w := inferenceTestNew(t, fixtureDevice, []int{2}, 1)
	bias := inferenceTestNew(t, fixtureDevice, []int{2}, 2)
	cw := inferenceTestNew(t, fixtureDevice, []int{2, 2, 1}, 3)
	for _, v := range []*Tensor{w, bias, cw} {
		if err := c.Bind(v); err != nil {
			t.Fatal(err)
		}
	}
	_, allocations, _ := inferenceFixture.snapshot()
	for _, op := range []func(){func() { GELU(x, true) }, func() { Softmax(x, -1) }, func() { LayerNorm(x, w, bias, 1e-5) }, func() { GroupNorm(x, w, bias, 1, 1e-5) }, func() { Conv1d(x, cw, bias, 1) }, func() { Concat(1, x, x) }, func() { ScaledDotProductAttention(x, x, x, nil) }} {
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, op)
	}
	if _, after, _ := inferenceFixture.snapshot(); after != allocations {
		t.Fatal("training rejection allocated buffers")
	}
	c.NoGrad(func() {
		assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { Dropout(x, .1, 1, true) })
		for _, op := range []func(){func() { Softmax(x, 3) }, func() { LayerNorm(x, w, bias, 0) }, func() { GroupNorm(x, w, bias, 3, 1e-5) }, func() { Concat(9, x, x) }, func() { Conv1dGEMMStride(x, cw, bias, 0, 0, 1) }} {
			assertGPUErrorPanic(t, gpu.ErrInvalidArgument, op)
		}
	})
	if _, after, _ := inferenceFixture.snapshot(); after != allocations {
		t.Fatal("invalid arguments allocated buffers")
	}
	for _, op := range []string{"gelu", "softmax", "norm", "conv", "attention", "concat"} {
		t.Run(op, func(t *testing.T) {
			live, _, _ := inferenceFixture.snapshot()
			inferenceFixture.mu.Lock()
			inferenceFixture.failNext = true
			inferenceFixture.mu.Unlock()
			c.NoGrad(func() {
				assertGPUErrorPanic(t, injectedInferenceError, func() {
					switch op {
					case "gelu":
						GELU(x, true)
					case "softmax":
						Softmax(x, 1)
					case "norm":
						LayerNorm(x, w, bias, 1e-5)
					case "conv":
						Conv1d(x, cw, bias, 1)
					case "attention":
						ScaledDotProductAttention(x, x, x, nil)
					case "concat":
						Concat(1, x, x)
					default:
						panic(fmt.Sprint(op))
					}
				})
			})
			if after, _, _ := inferenceFixture.snapshot(); after != live {
				t.Fatalf("failed %s leaked: %d -> %d", op, live, after)
			}
		})
	}
	c.NoGrad(func() {
		q := MulScalar(x, 1)
		defer q.ReleaseGraph()
		live, _, _ := inferenceFixture.snapshot()
		inferenceFixture.mu.Lock()
		inferenceFixture.failNext = true
		inferenceFixture.mu.Unlock()
		assertGPUErrorPanic(t, injectedInferenceError, func() { ScaledDotProductAttention(q, q, q, nil) })
		if after, _, _ := inferenceFixture.snapshot(); after != live {
			t.Fatal("failed attention leaked a local graph")
		}
		if _, err := q.ToHost(); err != nil {
			t.Fatalf("failed attention disposed caller input: %v", err)
		}
		for _, run := range []func() *Tensor{func() *Tensor { return GELU(x, false) }, func() *Tensor { return Softmax(x, 1) }, func() *Tensor { return LayerNorm(x, w, bias, 1e-5) }, func() *Tensor { return GroupNorm(x, w, bias, 1, 1e-5) }, func() *Tensor { return Conv1d(x, cw, bias, 1) }, func() *Tensor { return Concat(1, x, x) }, func() *Tensor { return ScaledDotProductAttention(x, x, x, nil) }} {
			y := run()
			if y.RequiresGrad || y.ExecutionContext() != c {
				t.Fatal("NoGrad/context propagation lost")
			}
			y.ReleaseGraph()
		}
		if err := c.SetOptions(ExecutionOptions{AttentionAlgorithm: "flash"}); err != nil {
			t.Fatal(err)
		}
		_, before, _ := inferenceFixture.snapshot()
		assertGPUErrorPanic(t, gpu.ErrUnsupported, func() { ScaledDotProductAttention(x, x, x, nil) })
		if _, after, _ := inferenceFixture.snapshot(); after != before {
			t.Fatal("unsupported FlashAttention allocated buffers")
		}
	})
}
