// Package autograd implements a define-by-run float32 tensor graph on CPU and CUDA.
package autograd

import (
	"errors"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"runtime"
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
func New(data []float32, shape []int, device tensor.Device, grad bool) (*Tensor, error) {
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
	for i := range values {
		values[i] = t.Data[storageIndex(i, t.Shape, t.Strides, t.Offset)]
	}
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
		s[i] = max(x, y)
	}
	return s
}
func bindex(flat int, out, src []int) int {
	os := strides(out)
	ss := strides(src)
	j := 0
	d := len(out) - len(src)
	for i := range src {
		c := (flat / os[i+d]) % out[i+d]
		if src[i] != 1 {
			j += c * ss[i]
		}
	}
	return j
}

type broadcastIndex struct{ d1, d2, s0, s1, s2, kind int }

func makeBroadcastIndex(out, src []int) broadcastIndex {
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
	}
	return (i/(q.d1*q.d2))*q.s0 + (i/q.d2%q.d1)*q.s1 + (i%q.d2)*q.s2
}
func parallelFor(n int, fn func(int, int)) {
	workers := min(runtime.GOMAXPROCS(0), n/65536)
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
func binary(a, b *Tensor, f, da, db func(float32, float32) float32) *Tensor {
	same(a, b)
	s := bshape(a.Shape, b.Shape)
	v := cpuAlloc(numel(s))
	if len(s) <= 3 {
		ai, bi := makeBroadcastIndex(s, a.Shape), makeBroadcastIndex(s, b.Shape)
		parallelFor(len(v), func(start, end int) {
			for i := start; i < end; i++ {
				v[i] = f(a.Data[ai.at(i)], b.Data[bi.at(i)])
			}
		})
	} else {
		for i := range v {
			v[i] = f(a.Data[bindex(i, s, a.Shape)], b.Data[bindex(i, s, b.Shape)])
		}
	}
	return result(v, s, []*Tensor{a, b}, func(g []float32) {
		ag := cpuAlloc(len(a.Data))
		bg := cpuAlloc(len(b.Data))
		if len(s) <= 3 {
			ai, bi := makeBroadcastIndex(s, a.Shape), makeBroadcastIndex(s, b.Shape)
			workers := min(runtime.GOMAXPROCS(0), len(g)/65536)
			if workers < 2 {
				workers = 1
			}
			localA := make([][]float32, workers)
			localB := make([][]float32, workers)
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				start, end := len(g)*w/workers, len(g)*(w+1)/workers
				wg.Add(1)
				go func(w, start, end int) {
					defer wg.Done()
					aa, bb := ag, bg
					if workers > 1 && len(ag) != len(g) {
						aa = make([]float32, len(ag))
						localA[w] = aa
					}
					if workers > 1 && len(bg) != len(g) {
						bb = make([]float32, len(bg))
						localB[w] = bb
					}
					for i := start; i < end; i++ {
						ia, ib := ai.at(i), bi.at(i)
						x, y := a.Data[ia], b.Data[ib]
						aa[ia] += g[i] * da(x, y)
						bb[ib] += g[i] * db(x, y)
					}
				}(w, start, end)
			}
			wg.Wait()
			for _, local := range localA {
				for i, v := range local {
					ag[i] += v
				}
			}
			for _, local := range localB {
				for i, v := range local {
					bg[i] += v
				}
			}
		} else {
			for i, z := range g {
				ia, ib := bindex(i, s, a.Shape), bindex(i, s, b.Shape)
				x, y := a.Data[ia], b.Data[ib]
				ag[ia] += z * da(x, y)
				bg[ib] += z * db(x, y)
			}
		}
		a.addGrad(ag)
		b.addGrad(bg)
		cpuRelease(ag)
		cpuRelease(bg)
	})
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
	b, _ := New([]float32{v}, []int{}, a.Device, false)
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
		dx := cpuAlloc(len(g))
		scale := float32(1)
		if op == 2 {
			scale = value
		} else if op == 3 {
			scale = 1 / value
		}
		parallelFor(len(g), func(start, end int) {
			for i := start; i < end; i++ {
				dx[i] = g[i] * scale
			}
		})
		a.addGrad(dx)
		cpuRelease(dx)
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
	for i, x := range a.Data {
		v[i] = f(x)
	}
	return result(v, append([]int(nil), a.Shape...), []*Tensor{a}, func(g []float32) {
		dx := cpuAlloc(len(g))
		for i, x := range a.Data {
			dx[i] = g[i] * d(x)
		}
		a.addGrad(dx)
		cpuRelease(dx)
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
