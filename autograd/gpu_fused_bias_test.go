package autograd

import (
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

// Disabled by default so existing tests retain the composition fallback.
func (b *inferenceTestBackend) BiasGELU(out, x, bias *gpu.Buffer, rows, cols int) error {
	if !b.fusedBiasSupported() {
		return gpu.ErrUnsupported
	}
	if rows == 0 {
		_, err := b.memory(out, 0)
		return err
	}
	return b.reference(out, []int{rows, cols}, []*gpu.Buffer{x, bias}, [][]int{{rows, cols}, {cols}}, func(v []*Tensor) *Tensor { return BiasGELU(v[0], v[1]) })
}
func (b *inferenceTestBackend) BiasResidual(out, x, bias, residual *gpu.Buffer, rows, cols int) error {
	if !b.fusedBiasSupported() {
		return gpu.ErrUnsupported
	}
	if rows == 0 {
		_, err := b.memory(out, 0)
		return err
	}
	return b.reference(out, []int{rows, cols}, []*gpu.Buffer{x, bias, residual}, [][]int{{rows, cols}, {cols}, {rows, cols}}, func(v []*Tensor) *Tensor { return BiasResidual(v[0], v[1], v[2]) })
}
func (b *inferenceTestBackend) fusedBiasSupported() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.fusedBiasCalls++
	return b.fusedBiasEnabled
}

func TestInferenceGPUFusedBiasCapability(t *testing.T) {
	b := inferenceFixture
	defer func() { b.mu.Lock(); b.fusedBiasEnabled = false; b.failNext = false; b.mu.Unlock() }()
	for _, enabled := range []bool{false, true} {
		for _, residual := range []bool{false, true} {
			t.Run(fmt.Sprintf("enabled=%t/residual=%t", enabled, residual), func(t *testing.T) {
				inferenceLeakCheck(t, fixtureDevice)
				x := inferenceTestNew(t, fixtureDevice, []int{2, 3}, 1)
				bias := inferenceTestNew(t, fixtureDevice, []int{3}, 2)
				b.mu.Lock()
				b.fusedBiasEnabled = enabled
				calls := b.fusedBiasCalls
				b.mu.Unlock()
				_, allocs, reads := b.snapshot()
				var y *Tensor
				if residual {
					y = BiasResidual(x, bias, x)
				} else {
					y = BiasGELU(x, bias)
				}
				defer y.ReleaseGraph()
				_, after, downloads := b.snapshot()
				b.mu.Lock()
				used := b.fusedBiasCalls - calls
				b.mu.Unlock()
				if used != 1 {
					t.Fatalf("fusion capability called %d times", used)
				}
				want := 4
				if enabled {
					want = 1
				}
				if after-allocs != want || downloads != reads {
					t.Fatalf("allocations=%d want=%d, downloads=%d", after-allocs, want, downloads-reads)
				}
				cpuX := inferenceTestNew(t, tensor.CPU, []int{2, 3}, 1)
				cpuBias := inferenceTestNew(t, tensor.CPU, []int{3}, 2)
				var ref *Tensor
				if residual {
					ref = BiasResidual(cpuX, cpuBias, cpuX)
				} else {
					ref = BiasGELU(cpuX, cpuBias)
				}
				defer ref.ReleaseGraph()
				got, err := y.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				expected, err := ref.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range got {
					if v != expected[i] {
						t.Fatalf("[%d]=%g want %g", i, v, expected[i])
					}
				}
			})
		}
	}
	b.mu.Lock()
	b.fusedBiasEnabled = true
	b.mu.Unlock()
	for _, residual := range []bool{false, true} {
		t.Run(fmt.Sprintf("failure/%t", residual), func(t *testing.T) {
			x := Transpose(inferenceTestNew(t, fixtureDevice, []int{3, 2}, 1), 0, 1)
			defer x.Close()
			bias := Slice(inferenceTestNew(t, fixtureDevice, []int{4}, 2), 0, 1, 4)
			defer bias.Close()
			live, _, _ := b.snapshot()
			b.mu.Lock()
			b.failNext = true
			b.mu.Unlock()
			assertGPUErrorPanic(t, injectedInferenceError, func() {
				if residual {
					BiasResidual(x, bias, x)
				} else {
					BiasGELU(x, bias)
				}
			})
			if after, _, _ := b.snapshot(); after != live {
				t.Fatal("fused failure leaked output or materialization scratch")
			}
			if _, err := x.ToHost(); err != nil {
				t.Fatal("fused failure disposed input", err)
			}
		})
	}
}

func TestInferenceGPUFusedBiasLayouts(t *testing.T) {
	inferenceFixture.mu.Lock()
	inferenceFixture.fusedBiasEnabled = true
	inferenceFixture.mu.Unlock()
	defer func() {
		inferenceFixture.mu.Lock()
		inferenceFixture.fusedBiasEnabled = false
		inferenceFixture.mu.Unlock()
	}()
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		for _, name := range []string{"matrix_bias", "residual_broadcast", "expanded", "offset", "rank6"} {
			t.Run(name, func(t *testing.T) {
				inferenceLeakCheck(t, device)
				build := func(d tensor.Device) *Tensor {
					x := inferenceTestNew(t, d, []int{2, 3}, 1)
					bias := inferenceTestNew(t, d, []int{3}, 2)
					switch name {
					case "matrix_bias":
						return BiasGELU(x, Reshape(bias, 1, 3))
					case "residual_broadcast":
						return BiasResidual(x, bias, inferenceTestNew(t, d, []int{1, 3}, 3))
					case "expanded":
						return BiasGELU(Expand(inferenceTestNew(t, d, []int{1, 3}, 1), 2, 3), bias)
					case "offset":
						return BiasResidual(Slice(inferenceTestNew(t, d, []int{2, 4}, 1), 1, 1, 4), Slice(inferenceTestNew(t, d, []int{4}, 2), 0, 1, 4), x)
					case "rank6":
						return BiasGELU(inferenceTestNew(t, d, []int{1, 2, 1, 2, 2, 3}, 1), bias)
					default:
						panic(name)
					}
				}
				got, ref := build(device), build(tensor.CPU)
				defer got.ReleaseGraph()
				defer ref.ReleaseGraph()
				a, err := got.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				want, err := ref.ToHost()
				if err != nil {
					t.Fatal(err)
				}
				if len(a) != len(want) {
					t.Fatal("output size mismatch")
				}
				for i, v := range a {
					if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 1e-5 {
						t.Fatalf("[%d]=%g want %g", i, v, want[i])
					}
				}
			})
		}
	})
}

func TestInferenceGPUFusedBiasEmpty(t *testing.T) {
	inferenceFixture.mu.Lock()
	inferenceFixture.fusedBiasEnabled = true
	inferenceFixture.mu.Unlock()
	defer func() {
		inferenceFixture.mu.Lock()
		inferenceFixture.fusedBiasEnabled = false
		inferenceFixture.mu.Unlock()
	}()
	inferenceTestDevices(t, func(t *testing.T, device tensor.Device) {
		backend, err := gpu.Get(string(device))
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := backend.(gpu.FusedBiasBackend); !ok {
			t.Skip("optional fusion capability not implemented")
		}
		inferenceLeakCheck(t, device)
		x := inferenceTestNew(t, device, []int{0, 3}, 1)
		bias := inferenceTestNew(t, device, []int{3}, 2)
		for _, run := range []func() *Tensor{func() *Tensor { return BiasGELU(x, bias) }, func() *Tensor { return BiasResidual(x, bias, x) }} {
			y := run()
			values, err := y.ToHost()
			y.ReleaseGraph()
			if err != nil || len(values) != 0 || len(y.Shape) != 2 || y.Shape[0] != 0 || y.Shape[1] != 3 {
				t.Fatalf("empty output: shape=%v values=%v err=%v", y.Shape, values, err)
			}
		}
	})
}
