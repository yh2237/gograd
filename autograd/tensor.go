// Package autograd implements a define-by-run float32 tensor graph on CPU and CUDA.
package autograd

import (
	"errors"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"
)

type Tensor struct {
	Data, Grad       []float32
	Shape, Strides   []int
	Offset           int
	DType            DType
	Device           tensor.Device
	buf              *cuda.Buffer
	storage          *tensorStorage
	bf16Storage      *tensorStorage
	bf16Version      uint64
	gradBuf          *cuda.Buffer
	aux              []*cuda.Buffer
	auxCPU           [][]float32
	savedLeases      []*tensorStorage
	ephemeral        bool
	RequiresGrad     bool
	execution        *ExecutionContext
	parents          []*Tensor
	gradParents      []*Tensor
	savedVersions    []uint64
	outputVersion    uint64
	backward         func([]float32)
	backwardGPU      func(*cuda.Buffer)
	retain           bool
	retainData       bool
	intermediate     bool
	historyReleased  bool
	children         atomic.Int64
	closed           atomic.Bool
	disposed         atomic.Bool
	releaseRequested atomic.Bool
	inBackward       atomic.Bool
}
type DType string

const Float32 DType = "float32"

func (t *Tensor) IsContiguous() bool {
	if t.Offset != 0 {
		return false
	}
	if t.Device == tensor.CPU && len(t.Data) != t.Numel() {
		return false
	}
	if t.storage != nil && t.storage.gpuBuffer != nil && t.storage.gpuBuffer.Len() != t.Numel() {
		return false
	}
	expected := strides(t.Shape)
	if len(expected) != len(t.Strides) {
		return false
	}
	for i := range expected {
		if expected[i] != t.Strides[i] {
			return false
		}
	}
	return true
}

var cpuPools sync.Map

func cpuAlloc(n int) []float32 {
	p, _ := cpuPools.LoadOrStore(n, &sync.Pool{})
	if v := p.(*sync.Pool).Get(); v != nil {
		out := v.([]float32)[:n]
		clear(out)
		return out
	}
	return make([]float32, n)
}
func cpuRelease(v []float32) {
	if v == nil {
		return
	}
	p, _ := cpuPools.LoadOrStore(len(v), &sync.Pool{})
	p.(*sync.Pool).Put(v)
}

// ReleaseGraph requests reclamation of this graph's intermediates. A node still
// used by another branch/view is reclaimed only after that dependent releases
// it. RetainData/RetainGrad pins a handle's data, not its derivative history.
// Persistent factory leaves (inputs/parameters) are closed by their callers.
func (t *Tensor) ReleaseGraph() {
	if t == nil || t.disposed.Load() {
		return
	}
	nodes := t.collectLifetime()
	for _, v := range nodes {
		if v.intermediate || v.ephemeral {
			v.releaseRequested.Store(true)
		}
	}
	for _, v := range nodes {
		v.maybeDispose()
	}
}

// NoGrad is the compatibility scope for unbound tensors. For independent
// execution use ExecutionContext.NoGrad with bound parameters and inputs.
func NoGrad(fn func()) { defaultExecutionContext.NoGrad(fn) }
func numel(s []int) int {
	n := 1
	for _, v := range s {
		n *= v
	}
	return n
}
func strides(s []int) []int {
	out := make([]int, len(s))
	n := 1
	for i := len(s) - 1; i >= 0; i-- {
		out[i] = n
		n *= s[i]
	}
	return out
}

// validateShape rejects dimensions that cannot describe storage. A negative
// dimension can otherwise cancel another negative dimension in numel, and a
// product that overflows int can wrap to a plausible element count.
func validateShape(shape []int) error {
	limit := int(^uint(0) >> 1)
	n := 1
	for _, d := range shape {
		if d < 0 {
			return errors.New("autograd: negative dimension")
		}
		if d != 0 && n > limit/d {
			return errors.New("autograd: oversized shape")
		}
		n *= d
	}
	return nil
}

func New(data []float32, shape []int, device tensor.Device, grad bool) (*Tensor, error) {
	if inferenceDevice(device) {
		return newInferenceTensor(data, shape, device, grad, false)
	}
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	if numel(shape) != len(data) {
		return nil, errors.New("autograd: shape/data mismatch")
	}
	if device == tensor.CUDA && !cuda.Available() {
		return nil, cuda.ErrUnavailable
	}
	if device != tensor.CPU && device != tensor.CUDA {
		return nil, errors.New("autograd: invalid device")
	}
	if device == tensor.CUDA {
		buf, err := cuda.Alloc(len(data) * 4)
		if err != nil {
			return nil, err
		}
		if err = buf.CopyFromHost(floatBytes(data)); err != nil {
			buf.Free()
			return nil, err
		}
		return &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, buf: buf, storage: deviceStorage(buf), RequiresGrad: grad}, nil
	}
	values := append([]float32(nil), data...)
	return &Tensor{Data: values, storage: cpuStorage(values, false), Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, RequiresGrad: grad}, nil
}
func floatBytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}
func (t *Tensor) Numel() int { return numel(t.Shape) }
func (t *Tensor) ToHost() ([]float32, error) {
	if err := t.checkOpen(); err != nil {
		return nil, err
	}
	if !t.IsContiguous() {
		c := t.Contiguous()
		v, e := c.ToHost()
		if c != t {
			c.Close()
		}
		return v, e
	}
	if t.Device == tensor.CPU {
		return append([]float32(nil), t.Data...), nil
	}
	if inferenceDevice(t.Device) {
		return t.storage.gpuBuffer.Backend().Download(t.storage.gpuBuffer, t.Numel())
	}
	v := make([]float32, t.Numel())
	if e := t.buf.CopyToHost(floatBytes(v)); e != nil {
		return nil, e
	}
	return v, nil
}
func (t *Tensor) GradToHost() ([]float32, error) {
	if err := t.checkOpen(); err != nil {
		return nil, err
	}
	if t.Device == tensor.CPU {
		return append([]float32(nil), t.Grad...), nil
	}
	if t.gradBuf == nil {
		return nil, nil
	}
	v := make([]float32, t.Numel())
	if e := t.gradBuf.CopyToHost(floatBytes(v)); e != nil {
		return nil, e
	}
	return v, nil
}
func (t *Tensor) CopyFrom(data []float32) error {
	if err := t.checkOpen(); err != nil {
		return err
	}
	if len(data) != t.Numel() {
		return errors.New("autograd: copy size mismatch")
	}
	if inferenceDevice(t.Device) {
		if !t.IsContiguous() {
			if err := copyInferenceView(t, data); err != nil {
				return err
			}
		} else {
			if err := t.storage.gpuBuffer.Backend().CopyFrom(t.storage.gpuBuffer, data); err != nil {
				return err
			}
		}
		t.storage.version.Add(1)
		return nil
	}
	if t.Device == tensor.CPU {
		if !t.IsContiguous() {
			for i, v := range data {
				t.Data[storageIndex(i, t.Shape, t.Strides, t.Offset)] = v
			}
			t.storage.version.Add(1)
			return nil
		}
		copy(t.Data, data)
		t.storage.version.Add(1)
		return nil
	}
	if !t.IsContiguous() {
		return errors.New("autograd: CopyFrom requires a contiguous CUDA tensor")
	}
	t.invalidateBF16()
	return t.buf.CopyFromHost(floatBytes(data))
}

// Buffer is borrowed; callers must not Free it or retain it after Close.
func (t *Tensor) Buffer() *cuda.Buffer {
	if t.closed.Load() {
		return nil
	}
	return t.buf
}

// Close ends this handle's lifetime. Dependencies still using its values or
// saved state keep them alive internally until their last edge is released.
// Repeated Close calls are safe.
func (t *Tensor) Close() {
	if t == nil {
		return
	}
	t.closed.Store(true)
	t.maybeDispose()
}
func Must(data []float32, shape []int, grad bool) *Tensor {
	v, e := New(data, shape, tensor.CPU, grad)
	if e != nil {
		panic(e)
	}
	return v
}
func Zeros(shape []int, device tensor.Device, grad bool) (*Tensor, error) {
	if inferenceDevice(device) {
		return newInferenceTensor(nil, shape, device, grad, true)
	}
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	if device == tensor.CUDA {
		buf, e := cuda.Alloc(numel(shape) * 4)
		if e != nil {
			return nil, e
		}
		if e = buf.Memset(0, buf.Size()); e != nil {
			buf.Free()
			return nil, e
		}
		return &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, buf: buf, storage: deviceStorage(buf), RequiresGrad: grad}, nil
	}
	return New(make([]float32, numel(shape)), shape, device, grad)
}
func (t *Tensor) Detach() *Tensor {
	t.assertOpen()
	if inferenceDevice(t.Device) {
		return detachInference(t)
	}
	if t.Device == tensor.CUDA {
		src := t
		if !t.IsContiguous() {
			src = t.Contiguous()
			defer src.Close()
		}
		v, e := New(make([]float32, t.Numel()), t.Shape, t.Device, false)
		if e != nil {
			panic(e)
		}
		if e = copyDevice(v.buf, src.buf, t.Numel()); e != nil {
			v.Close()
			panic(e)
		}
		v.execution = t.execution
		return v
	}
	values := make([]float32, t.Numel())
	gatherView(values, t.Data, t.Shape, t.Strides, t.Offset)
	return &Tensor{Data: values, storage: cpuStorage(values, false), Shape: append([]int(nil), t.Shape...), Strides: strides(t.Shape), DType: t.DType, Device: t.Device, execution: t.execution}
}
func (t *Tensor) RetainGrad() { t.assertOpen(); t.retain = true }
func (t *Tensor) ZeroGrad() {
	t.assertOpen()
	t.Grad = nil
	if t.gradBuf != nil {
		if t.Device == tensor.CUDA && !t.intermediate && t.RequiresGrad {
			if e := t.gradBuf.Memset(0, t.gradBuf.Size()); e != nil {
				panic(e)
			}
		} else {
			t.gradBuf.Free()
			t.gradBuf = nil
		}
	}
}
func result(data []float32, shape []int, parents []*Tensor, back func([]float32)) *Tensor {
	return resultWithSaved(data, shape, parents, nil, back)
}

func resultWithSaved(data []float32, shape []int, parents, saved []*Tensor, back func([]float32)) *Tensor {
	context, req := graphRecording(parents, saved...)
	if !req {
		back = nil
	}
	// Keep lifetime parents even when derivatives are disabled. ReleaseGraph
	// needs them to reclaim inference intermediates and shared-storage views.
	r := &Tensor{Data: data, storage: cpuStorage(data, true), Shape: shape, Strides: strides(shape), DType: Float32, Device: parents[0].Device, RequiresGrad: req, execution: context, backward: back, intermediate: true}
	r.attachParents(parents)
	for _, p := range saved {
		r.keepAlive(p)
	}
	return r
}
func resultGPU(buf *cuda.Buffer, shape []int, parents []*Tensor, back func(*cuda.Buffer)) *Tensor {
	return resultGPUWithSaved(buf, shape, parents, nil, back)
}

func resultGPUWithSaved(buf *cuda.Buffer, shape []int, parents, saved []*Tensor, back func(*cuda.Buffer)) *Tensor {
	context, req := graphRecording(parents, saved...)
	if !req {
		back = nil
	}
	r := &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: tensor.CUDA, buf: buf, storage: deviceStorage(buf), RequiresGrad: req, execution: context, backwardGPU: back, intermediate: true}
	r.attachParents(parents)
	for _, p := range saved {
		r.keepAlive(p)
	}
	return r
}
func (t *Tensor) ensureGradGPU() *cuda.Buffer {
	if !t.RequiresGrad {
		return nil
	}
	if t.gradBuf == nil {
		b, e := cuda.Alloc(t.Numel() * 4)
		if e != nil {
			panic(e)
		}
		if e = b.Memset(0, b.Size()); e != nil {
			panic(e)
		}
		t.gradBuf = b
	}
	return t.gradBuf
}
func (t *Tensor) addGrad(g []float32) {
	if !t.RequiresGrad {
		return
	}
	if t.Grad == nil {
		t.Grad = make([]float32, t.Numel())
	}
	for i, v := range g {
		t.Grad[i] += v
	}
}
func (t *Tensor) Backward() error {
	return t.BackwardWithOptions(BackwardOptions{})
}
func same(a *Tensor, others ...*Tensor) {
	a.assertOpen()
	context := a.execution
	for _, b := range others {
		if b == nil {
			continue
		}
		b.assertOpen()
		if a.Device != b.Device {
			panic("autograd: mixed devices")
		}
		context = mergeExecutionContext(context, b)
	}
}
func bshape(a, b []int) []int {
	n := max(len(a), len(b))
	s := make([]int, n)
	for i := 0; i < n; i++ {
		x, y := 1, 1
		if i >= n-len(a) {
			x = a[i-(n-len(a))]
		}
		if i >= n-len(b) {
			y = b[i-(n-len(b))]
		}
		if x != y && x != 1 && y != 1 {
			panic("autograd: broadcast mismatch")
		}
		// A size-one dim adopts the other length, so a zero dim stays zero
		// instead of becoming one.
		if x == 1 {
			s[i] = y
		} else {
			s[i] = x
		}
	}
	return s
}
func bindex(flat int, out, src []int) int {
	return bindexStrided(flat, out, strides(out), src, strides(src))
}

// bindexStrided is bindex with the row-major strides hoisted out of the loop
// that walks the batch axis.
func bindexStrided(flat int, out, outStrides, src, srcStrides []int) int {
	j := 0
	d := len(out) - len(src)
	for i := range src {
		c := (flat / outStrides[i+d]) % out[i+d]
		if src[i] != 1 {
			j += c * srcStrides[i]
		}
	}
	return j
}

// broadcastIndex maps an output element to one operand. Ranks up to three use
// the packed fields; higher ranks keep the stride vectors and walk only the
// dims the operand actually varies along.
type broadcastIndex struct {
	d1, d2, s0, s1, s2, kind int
	out, outStrides          []int
	src, srcStrides          []int
}

func makeBroadcastIndex(out, src []int) broadcastIndex {
	if len(out) > 3 {
		q := broadcastIndex{kind: 5, out: out, outStrides: strides(out), src: src, srcStrides: strides(src)}
		if slices.Equal(out, src) {
			q.kind = 1
		} else if numel(src) == 1 {
			q.kind = 2
		}
		return q
	}
	o := [3]int{1, 1, 1}
	v := [3]int{1, 1, 1}
	for i, x := range out {
		o[3-len(out)+i] = x
	}
	for i, x := range src {
		v[3-len(src)+i] = x
	}
	q := broadcastIndex{d1: o[1], d2: o[2]}
	if len(out) == len(src) {
		equal := true
		for i := range out {
			if out[i] != src[i] {
				equal = false
				break
			}
		}
		if equal {
			q.kind = 1
			return q
		}
	}
	if numel(src) == 1 {
		q.kind = 2
		return q
	}
	if len(src) == 1 && src[0] == out[len(out)-1] {
		q.kind = 3
		return q
	}
	if len(out) == 3 && len(src) == 3 && src[0] == out[0] && src[1] == 1 && src[2] == out[2] {
		q.kind = 4
		return q
	}
	if v[0] != 1 {
		q.s0 = v[1] * v[2]
	}
	if v[1] != 1 {
		q.s1 = v[2]
	}
	if v[2] != 1 {
		q.s2 = 1
	}
	return q
}
func (q broadcastIndex) at(i int) int {
	switch q.kind {
	case 1:
		return i
	case 2:
		return 0
	case 3:
		return i % q.d2
	case 4:
		return i/(q.d1*q.d2)*q.d2 + i%q.d2
	case 5:
		j := 0
		d := len(q.out) - len(q.src)
		for k := range q.src {
			if q.src[k] == 1 {
				continue
			}
			j += ((i / q.outStrides[k+d]) % q.out[k+d]) * q.srcStrides[k]
		}
		return j
	}
	return (i/(q.d1*q.d2))*q.s0 + (i/q.d2%q.d1)*q.s1 + (i%q.d2)*q.s2
}
func parallelFor(n int, fn func(int, int)) {
	parallelForWork(n, 1, fn)
}

// Split independent items using their element count, not just their count.
// A softmax row, for example, can contain hundreds of elements.
func parallelForWork(n, elementsPerItem int, fn func(int, int)) {
	workers := min(runtime.GOMAXPROCS(0), n, n*elementsPerItem/65536)
	if workers < 2 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start, end := n*w/workers, n*(w+1)/workers
		wg.Add(1)
		go func() { defer wg.Done(); fn(start, end) }()
	}
	wg.Wait()
}
func Add(a, b *Tensor) *Tensor {
	return dispatchElementwise("add", a, b)
}
func Sub(a, b *Tensor) *Tensor {
	return dispatchElementwise("sub", a, b)
}
func Mul(a, b *Tensor) *Tensor {
	return dispatchElementwise("mul", a, b)
}
func Div(a, b *Tensor) *Tensor {
	return dispatchElementwise("div", a, b)
}
func Scalar(a *Tensor, v float32) *Tensor {
	dispatchBackend("scalar", a.Device)
	b, err := New([]float32{v}, []int{}, a.Device, false)
	if err != nil {
		panic(err)
	}
	b.ephemeral = true
	b.execution = a.execution
	return b
}
func cpuScalar(a *Tensor, value float32, op int) *Tensor {
	out := cpuAlloc(a.Numel())
	parallelFor(len(out), func(start, end int) {
		for i := start; i < end; i++ {
			x := a.Data[i]
			switch op {
			case 0:
				out[i] = x + value
			case 1:
				out[i] = x - value
			case 2:
				out[i] = x * value
			case 3:
				out[i] = x / value
			}
		}
	})
	return result(out, a.Shape, []*Tensor{a}, func(g []float32) {
		if !a.RequiresGrad {
			return
		}
		if a.Grad == nil {
			a.Grad = make([]float32, a.Numel())
		}
		scale := float32(1)
		if op == 2 {
			scale = value
		} else if op == 3 {
			scale = 1 / value
		}
		grad := a.Grad
		parallelFor(len(g), func(start, end int) {
			for i := start; i < end; i++ {
				grad[i] += g[i] * scale
			}
		})
	})
}
func AddScalar(a *Tensor, v float32) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("add_scalar", a.Device) {
		return gpuScalar(a, v, 0)
	}
	return cpuScalar(a, v, 0)
}
func SubScalar(a *Tensor, v float32) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("sub_scalar", a.Device) {
		return gpuScalar(a, v, 1)
	}
	return cpuScalar(a, v, 1)
}
func MulScalar(a *Tensor, v float32) *Tensor {
	if inferenceDevice(a.Device) {
		return inferenceMulScalar(a, v)
	}
	a = a.Contiguous()
	if dispatchBackend("mul_scalar", a.Device) {
		return gpuScalar(a, v, 2)
	}
	return cpuScalar(a, v, 2)
}
func DivScalar(a *Tensor, v float32) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("div_scalar", a.Device) {
		return gpuScalar(a, v, 3)
	}
	return cpuScalar(a, v, 3)
}
func unary(a *Tensor, f, d func(float32) float32) *Tensor {
	v := cpuAlloc(len(a.Data))
	parallelFor(len(v), func(start, end int) {
		for i := start; i < end; i++ {
			v[i] = f(a.Data[i])
		}
	})
	return result(v, append([]int(nil), a.Shape...), []*Tensor{a}, func(g []float32) {
		if !a.RequiresGrad {
			return
		}
		if a.Grad == nil {
			a.Grad = make([]float32, a.Numel())
		}
		grad := a.Grad
		parallelFor(len(g), func(start, end int) {
			for i := start; i < end; i++ {
				grad[i] += g[i] * d(a.Data[i])
			}
		})
	})
}
func Exp(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("exp", a.Device) {
		return gpuUnary(a, 0)
	}
	return unary(a, func(x float32) float32 { return float32(math.Exp(float64(x))) }, func(x float32) float32 { return float32(math.Exp(float64(x))) })
}
func Log(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("log", a.Device) {
		return gpuUnary(a, 1)
	}
	return unary(a, func(x float32) float32 { return float32(math.Log(float64(x))) }, func(x float32) float32 { return 1 / x })
}
func Abs(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("abs", a.Device) {
		return gpuUnary(a, 2)
	}
	return unary(a, func(x float32) float32 { return float32(math.Abs(float64(x))) }, func(x float32) float32 {
		if x < 0 {
			return -1
		}
		if x > 0 {
			return 1
		}
		return 0
	})
}
func Tanh(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("tanh", a.Device) {
		return gpuUnary(a, 3)
	}
	return unary(a, func(x float32) float32 { return float32(math.Tanh(float64(x))) }, func(x float32) float32 { v := float32(math.Tanh(float64(x))); return 1 - v*v })
}

// Sin applies sine elementwise.
func Sin(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("sin", a.Device) {
		return gpuUnary(a, 7)
	}
	return unary(a, func(x float32) float32 { return float32(math.Sin(float64(x))) }, func(x float32) float32 { return float32(math.Cos(float64(x))) })
}

// Sqrt applies the square root elementwise. The derivative is infinite at zero,
// matching the reference.
func Sqrt(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("sqrt", a.Device) {
		return gpuUnary(a, 8)
	}
	return unary(a, func(x float32) float32 { return float32(math.Sqrt(float64(x))) }, func(x float32) float32 {
		return .5 / float32(math.Sqrt(float64(x)))
	})
}

// Sigmoid applies the logistic function elementwise.
func Sigmoid(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("sigmoid", a.Device) {
		return gpuUnary(a, 9)
	}
	return unary(a, func(x float32) float32 { return sigmoid32(x) }, func(x float32) float32 {
		s := sigmoid32(x)
		return s * (1 - s)
	})
}

func sigmoid32(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// Clamp limits each element to [lo, hi]. The gradient passes through where the
// input lies inside the interval, endpoints included.
func Clamp(a *Tensor, lo, hi float32) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("clamp", a.Device) {
		return gpuClamp(a, lo, hi)
	}
	return unary(a, func(x float32) float32 { return min(max(x, lo), hi) }, func(x float32) float32 {
		if x >= lo && x <= hi {
			return 1
		}
		return 0
	})
}

// PowScalar raises each element to a constant power.
func PowScalar(a *Tensor, p float32) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("pow_scalar", a.Device) {
		return gpuPowScalar(a, p)
	}
	return unary(a, func(x float32) float32 { return float32(math.Pow(float64(x), float64(p))) }, func(x float32) float32 {
		if p == 0 {
			return 0
		}
		return p * float32(math.Pow(float64(x), float64(p-1)))
	})
}
func ReLU(a *Tensor) *Tensor {
	a = a.Contiguous()
	if dispatchBackend("relu", a.Device) {
		return gpuUnary(a, 4)
	}
	return unary(a, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	}, func(x float32) float32 {
		if x > 0 {
			return 1
		}
		return 0
	})
}
func GELU(a *Tensor, approx bool) *Tensor {
	if inferenceDevice(a.Device) {
		return inferenceGELU(a, approx)
	}
	a = a.Contiguous()
	if dispatchBackend("gelu", a.Device) {
		if approx {
			return gpuUnary(a, 6)
		}
		return gpuUnary(a, 5)
	}
	if !approx {
		return unary(a, func(x float32) float32 { v := float64(x); return float32(.5 * v * (1 + math.Erf(v/math.Sqrt2))) }, func(x float32) float32 {
			v := float64(x)
			return float32(.5*(1+math.Erf(v/math.Sqrt2)) + v*math.Exp(-v*v/2)/math.Sqrt(2*math.Pi))
		})
	}
	return unary(a, func(x float32) float32 {
		v := float64(x)
		return float32(.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+.044715*v*v*v))))
	}, func(x float32) float32 {
		v := float64(x)
		u := math.Sqrt(2/math.Pi) * (v + .044715*v*v*v)
		th := math.Tanh(u)
		return float32(.5*(1+th) + .5*v*(1-th*th)*math.Sqrt(2/math.Pi)*(1+3*.044715*v*v))
	})
}
