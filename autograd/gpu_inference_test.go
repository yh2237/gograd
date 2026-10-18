package autograd

import (
	"errors"
	"os"
	"sync"
	"testing"

	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	"github.com/yh2237/gograd/vulkan"
)

// Test-only storage/dispatch fixture. This is deliberately not a production
// backend or a CPU fallback: it verifies framework ownership without a driver.
type inferenceTestMemory struct{ data []float32 }
type inferenceTestBackend struct {
	gpu.UnsupportedOps
	mu                           sync.Mutex
	live, allocations, downloads int
	failNext                     bool
	failCopy, failZero           bool
}

var inferenceFixture = &inferenceTestBackend{}
var injectedInferenceError = errors.New("injected inference operation failure")

const fixtureDevice tensor.Device = "testgpu"

func init() {
	if err := gpu.Register(inferenceFixture); err != nil {
		panic(err)
	}
}
func (*inferenceTestBackend) Name() string    { return string(fixtureDevice) }
func (*inferenceTestBackend) Available() bool { return true }
func (b *inferenceTestBackend) snapshot() (int, int, int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.live, b.allocations, b.downloads
}
func (b *inferenceTestBackend) Alloc(n int) (*gpu.Buffer, error) {
	if _, err := gpu.Elements(n); err != nil {
		return nil, err
	}
	m := &inferenceTestMemory{data: make([]float32, n)}
	b.mu.Lock()
	b.live++
	b.allocations++
	b.mu.Unlock()
	return gpu.NewBuffer(b, n, m, func() { b.mu.Lock(); b.live--; b.mu.Unlock(); m.data = nil })
}
func (b *inferenceTestBackend) memory(x *gpu.Buffer, n int) (*inferenceTestMemory, error) {
	if err := gpu.CheckBuffer(b, x, n); err != nil {
		return nil, err
	}
	native, err := x.Native(b)
	if err != nil {
		return nil, err
	}
	return native.(*inferenceTestMemory), nil
}
func (b *inferenceTestBackend) Upload(v []float32) (*gpu.Buffer, error) {
	x, err := b.Alloc(len(v))
	if err != nil {
		return nil, err
	}
	if err := b.CopyFrom(x, v); err != nil {
		x.Close()
		return nil, err
	}
	return x, nil
}
func (b *inferenceTestBackend) CopyFrom(x *gpu.Buffer, v []float32) error {
	m, err := b.memory(x, len(v))
	if err != nil {
		return err
	}
	copy(m.data, v)
	return nil
}
func (b *inferenceTestBackend) Download(x *gpu.Buffer, n int) ([]float32, error) {
	m, err := b.memory(x, x.Len())
	if err != nil {
		return nil, err
	}
	if n < 0 || n > x.Len() {
		return nil, gpu.ErrInvalidArgument
	}
	b.mu.Lock()
	b.downloads++
	b.mu.Unlock()
	return append([]float32(nil), m.data[:n]...), nil
}
func (b *inferenceTestBackend) Zero(x *gpu.Buffer) error {
	b.mu.Lock()
	fail := b.failZero
	b.failZero = false
	b.mu.Unlock()
	if fail {
		return injectedInferenceError
	}
	m, err := b.memory(x, x.Len())
	if err != nil {
		return err
	}
	clear(m.data)
	return nil
}
func (b *inferenceTestBackend) Release(x *gpu.Buffer) { x.Close() }
func (b *inferenceTestBackend) Copy(out, src *gpu.Buffer, n int) error {
	b.mu.Lock()
	fail := b.failCopy
	b.failCopy = false
	b.mu.Unlock()
	if fail {
		panic(injectedInferenceError)
	}
	x, err := b.memory(src, n)
	if err != nil {
		return err
	}
	y, err := b.memory(out, n)
	if err != nil {
		return err
	}
	copy(y.data, x.data)
	return nil
}
func (b *inferenceTestBackend) binary(out, a, right *gpu.Buffer, n, op int) error {
	x, err := b.memory(a, n)
	if err != nil {
		return err
	}
	y, err := b.memory(right, n)
	if err != nil {
		return err
	}
	z, err := b.memory(out, n)
	if err != nil {
		return err
	}
	b.mu.Lock()
	fail := b.failNext
	b.failNext = false
	b.mu.Unlock()
	if fail {
		return injectedInferenceError
	}
	for i := range z.data {
		if op == 0 {
			z.data[i] = x.data[i] + y.data[i]
		} else {
			z.data[i] = x.data[i] * y.data[i]
		}
	}
	return nil
}
func (b *inferenceTestBackend) Add(out, a, right *gpu.Buffer, n int) error {
	return b.binary(out, a, right, n, 0)
}
func (b *inferenceTestBackend) Mul(out, a, right *gpu.Buffer, n int) error {
	return b.binary(out, a, right, n, 1)
}
func (b *inferenceTestBackend) MulScalar(out, a *gpu.Buffer, n int, scalar float32) error {
	x, err := b.memory(a, n)
	if err != nil {
		return err
	}
	y, err := b.memory(out, n)
	if err != nil {
		return err
	}
	for i := range y.data {
		y.data[i] = x.data[i] * scalar
	}
	return nil
}

func TestInferenceGPUStorageLifetime(t *testing.T) {
	before, _, _ := inferenceFixture.snapshot()
	c := NewExecutionContext()
	x, err := c.New([]float32{1, 2, 3, 4}, []int{2, 2}, fixtureDevice, true)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(x.Close)
	if x.Data != nil || x.Buffer() != nil || !x.RequiresGrad {
		t.Fatal("GPU factory staged data on CPU or lost RequiresGrad")
	}
	_, allocations, _ := inferenceFixture.snapshot()
	assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { Add(x, x) })
	assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { Reshape(x, 4) })
	if _, after, _ := inferenceFixture.snapshot(); after != allocations {
		t.Fatal("recording rejection allocated memory")
	}
	var view *Tensor
	c.NoGrad(func() { view = Reshape(MulScalar(x, 2), 4) })
	t.Cleanup(view.Close)
	if view.RequiresGrad || view.ExecutionContext() != c {
		t.Fatal("inference context/recording not preserved")
	}
	view.RetainData()
	x.Close()
	view.ReleaseGraph()
	host, err := view.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "retained view", host, []float32{2, 4, 6, 8}, 0, 0)
	alias := view.storage
	if err := view.CopyFrom([]float32{8, 6, 4, 2}); err != nil {
		t.Fatal(err)
	}
	if alias.version.Load() != 1 {
		t.Fatal("GPU CopyFrom did not increment shared version")
	}
	detached := view.Detach()
	t.Cleanup(detached.Close)
	if detached.ExecutionContext() != c || detached.RequiresGrad {
		t.Fatal("detach metadata")
	}
	view.Close()
	if err := detached.CopyFrom([]float32{1, 3, 5, 7}); err != nil {
		t.Fatal(err)
	}
	host, err = detached.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "detached", host, []float32{1, 3, 5, 7}, 0, 0)
	detached.Close()
	detached.Close()
	if live, _, _ := inferenceFixture.snapshot(); live != before {
		t.Fatalf("GPU storage leak: %d -> %d", before, live)
	}
}

func TestInferenceGPUDispatchAndFailure(t *testing.T) {
	c := NewExecutionContext()
	x, err := c.New([]float32{1, 2, 3}, []int{3}, fixtureDevice, false)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	_, _, downloads := inferenceFixture.snapshot()
	y := MulScalar(Mul(Add(x, x), x), .5)
	defer y.ReleaseGraph()
	if _, _, after := inferenceFixture.snapshot(); after != downloads {
		t.Fatal("forward downloaded values")
	}
	host, err := y.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "forward", host, []float32{1, 4, 9}, 0, 0)
	assertGPUErrorPanic(t, gpu.ErrUnsupported, func() { Sub(x, x) })
	assertGPUErrorPanic(t, gpu.ErrUnsupported, func() { Log(x) })
	live, _, _ := inferenceFixture.snapshot()
	inferenceFixture.mu.Lock()
	inferenceFixture.failNext = true
	inferenceFixture.mu.Unlock()
	assertGPUErrorPanic(t, injectedInferenceError, func() { c.NoGrad(func() { Add(x, x) }) })
	if after, _, _ := inferenceFixture.snapshot(); after != live {
		t.Fatal("failed forward leaked output")
	}
	inferenceFixture.mu.Lock()
	inferenceFixture.failCopy = true
	inferenceFixture.mu.Unlock()
	assertGPUErrorPanic(t, injectedInferenceError, func() { x.Detach() })
	if after, _, _ := inferenceFixture.snapshot(); after != live {
		t.Fatal("failed detach leaked output")
	}
	inferenceFixture.mu.Lock()
	inferenceFixture.failZero = true
	inferenceFixture.mu.Unlock()
	if _, err := c.Zeros([]int{3}, fixtureDevice, false); !errors.Is(err, injectedInferenceError) {
		t.Fatal(err)
	}
	if after, _, _ := inferenceFixture.snapshot(); after != live {
		t.Fatal("failed zero factory leaked output")
	}
	if !c.Recording() {
		t.Fatal("failed NoGrad scope not restored")
	}
	z, err := c.Zeros([]int{3}, fixtureDevice, false)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	host, err = z.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "zeros", host, []float32{0, 0, 0}, 0, 0)
	if err := z.Backward(); !errors.Is(err, gpu.ErrInferenceOnly) {
		t.Fatal(err)
	}
	assertGPUErrorPanic(t, gpu.ErrInferenceOnly, func() { c.NoGrad(func() { BatchNorm(x, nil, nil, nil, nil, BatchNormOptions{Training: true}) }) })
}

func TestVulkanTensorInference(t *testing.T) {
	if !vulkan.Available() {
		if os.Getenv("GOGRAD_REQUIRE_VULKAN") == "1" {
			t.Fatal(vulkan.Err())
		}
		t.Skipf("Vulkan unavailable: %v", vulkan.Err())
	}
	before := vulkan.MemoryStats().LiveBytes
	c := NewExecutionContext()
	a, b := make([]float32, 513), make([]float32, 513)
	for i := range a {
		a[i] = float32(i%17-8) / 16
		b[i] = float32(i%13-6) / 8
	}
	x, err := c.New(a, []int{3, 171}, tensor.Vulkan, true)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	w, err := c.New(b, []int{3, 171}, tensor.Vulkan, true)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	var y *Tensor
	c.NoGrad(func() { y = MulScalar(Add(Mul(x, w), x), .25) })
	defer y.ReleaseGraph()
	if y.Data != nil || y.Buffer() != nil || y.RequiresGrad {
		t.Fatal("Vulkan output is not an inference device tensor")
	}
	x.Close()
	w.Close()
	host, err := y.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(a))
	for i := range want {
		want[i] = (a[i]*b[i] + a[i]) * .25
	}
	conv2dClose(t, "Vulkan resident chain", host, want, 1e-5, 1e-5)
	detached := y.Detach()
	defer detached.Close()
	y.ReleaseGraph()
	host, err = detached.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "Vulkan detach after release", host, want, 1e-5, 1e-5)
	detached.Close()
	if after := vulkan.MemoryStats().LiveBytes; after != before {
		t.Fatalf("Vulkan graph leaked: %d -> %d", before, after)
	}
}
