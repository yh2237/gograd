// Package autograd implements a define-by-run float32 tensor graph. The CUDA
// device currently uses host staging for operators without a device kernel.
package autograd

import (
	"errors"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

type Tensor struct {
	Data, Grad     []float32
	Shape, Strides []int
	DType          DType
	Device         tensor.Device
	buf            *cuda.Buffer
	gradBuf        *cuda.Buffer
	aux            []*cuda.Buffer
	ownsBuffer     bool
	ownsData       bool
	ephemeral      bool
	RequiresGrad   bool
	parents        []*Tensor
	backward       func([]float32)
	backwardGPU    func(*cuda.Buffer)
	retain         bool
}
type DType string

const Float32 DType = "float32"

func (t *Tensor) IsContiguous() bool {
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

var recording = true
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

// ReleaseGraph returns temporary activations after Backward or inference.
// Callers must finish reading the graph's outputs first.
func (t *Tensor) ReleaseGraph() {
	seen := map[*Tensor]bool{}
	var walk func(*Tensor)
	walk = func(v *Tensor) {
		if seen[v] {
			return
		}
		seen[v] = true
		for _, p := range v.parents {
			walk(p)
		}
		if v.Device == tensor.CPU && v.ownsData && !v.retain {
			cpuRelease(v.Data)
			v.Data = nil
			v.ownsData = false
		}
		if v.Device == tensor.CUDA && ((len(v.parents) > 0 && !v.retain) || v.ephemeral) {
			v.Close()
		}
	}
	walk(t)
}

func NoGrad(fn func()) { old := recording; recording = false; defer func() { recording = old }(); fn() }
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
		return &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, buf: buf, ownsBuffer: true, RequiresGrad: grad}, nil
	}
	return &Tensor{Data: append([]float32(nil), data...), Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, RequiresGrad: grad}, nil
}
func floatBytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}
func (t *Tensor) Numel() int { return numel(t.Shape) }
func (t *Tensor) ToHost() ([]float32, error) {
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
	if len(data) != t.Numel() {
		return errors.New("autograd: copy size mismatch")
	}
	if t.Device == tensor.CPU {
		copy(t.Data, data)
		return nil
	}
	return t.buf.CopyFromHost(floatBytes(data))
}
func (t *Tensor) Buffer() *cuda.Buffer { return t.buf }
func (t *Tensor) Close() {
	for _, b := range t.aux {
		b.Free()
	}
	t.aux = nil
	if t.gradBuf != nil {
		t.gradBuf.Free()
		t.gradBuf = nil
	}
	if t.ownsBuffer && t.buf != nil {
		t.buf.Free()
		t.buf = nil
	}
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
		return &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, buf: buf, ownsBuffer: true, RequiresGrad: grad}, nil
	}
	return New(make([]float32, numel(shape)), shape, device, grad)
}
func (t *Tensor) Detach() *Tensor {
	if t.Device == tensor.CUDA {
		v, e := New(make([]float32, t.Numel()), t.Shape, t.Device, false)
		if e != nil {
			panic(e)
		}
		if e = copyDevice(v.buf, t.buf, t.Numel()); e != nil {
			panic(e)
		}
		return v
	}
	v := *t
	v.Data = append([]float32(nil), t.Data...)
	v.Grad = nil
	v.RequiresGrad = false
	v.parents = nil
	v.backward = nil
	return &v
}
func (t *Tensor) RetainGrad() { t.retain = true }
func (t *Tensor) ZeroGrad() {
	t.Grad = nil
	if t.gradBuf != nil {
		if t.Device == tensor.CUDA && len(t.parents) == 0 && t.RequiresGrad {
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
	req := false
	for _, p := range parents {
		req = req || p.RequiresGrad
	}
	return &Tensor{Data: data, Shape: shape, Strides: strides(shape), DType: Float32, Device: parents[0].Device, RequiresGrad: req && recording, parents: parents, backward: back, ownsData: true}
}
func resultGPU(buf *cuda.Buffer, shape []int, parents []*Tensor, back func(*cuda.Buffer)) *Tensor {
	req := false
	for _, p := range parents {
		req = req || p.RequiresGrad
	}
	return &Tensor{Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: tensor.CUDA, buf: buf, ownsBuffer: true, RequiresGrad: req && recording, parents: parents, backwardGPU: back}
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
		t.Grad = make([]float32, len(t.Data))
	}
	for i, v := range g {
		t.Grad[i] += v
	}
}
func (t *Tensor) Backward() error {
	if t.Numel() != 1 {
		return errors.New("autograd: backward requires scalar")
	}
	topo := []*Tensor{}
	seen := map[*Tensor]bool{}
	var visit func(*Tensor)
	visit = func(v *Tensor) {
		if seen[v] {
			return
		}
		seen[v] = true
		for _, p := range v.parents {
			visit(p)
		}
		topo = append(topo, v)
	}
	visit(t)
	if t.Device == tensor.CUDA {
		setOne(t.ensureGradGPU())
	} else {
		t.addGrad([]float32{1})
	}
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		if v.Device == tensor.CUDA {
			if v.backwardGPU != nil && v.gradBuf != nil {
				v.backwardGPU(v.gradBuf)
			}
			if (len(v.parents) > 0 && !v.retain) || v.ephemeral {
				v.Close()
			}
			continue
		}
		if v.backward != nil && v.Grad != nil {
			v.backward(v.Grad)
		}
		if len(v.parents) > 0 && !v.retain {
			v.Grad = nil
		}
	}
	return nil
}
func same(a, b *Tensor) {
	if a.Device != b.Device {
		panic("autograd: mixed devices")
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

type broadcastIndex struct{ d1, d2, s0, s1, s2 int }

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
	if a.Device == tensor.CUDA {
		same(a, b)
		return gpuBinary(a, b, 0)
	}
	return binary(a, b, func(x, y float32) float32 { return x + y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return 1 })
}
func Sub(a, b *Tensor) *Tensor {
	if a.Device == tensor.CUDA {
		same(a, b)
		return gpuBinary(a, b, 1)
	}
	return binary(a, b, func(x, y float32) float32 { return x - y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return -1 })
}
func Mul(a, b *Tensor) *Tensor {
	if a.Device == tensor.CUDA {
		same(a, b)
		return gpuBinary(a, b, 2)
	}
	return binary(a, b, func(x, y float32) float32 { return x * y }, func(x, y float32) float32 { return y }, func(x, y float32) float32 { return x })
}
func Div(a, b *Tensor) *Tensor {
	if a.Device == tensor.CUDA {
		same(a, b)
		return gpuBinary(a, b, 3)
	}
	return binary(a, b, func(x, y float32) float32 { return x / y }, func(x, y float32) float32 { return 1 / y }, func(x, y float32) float32 { return -x / (y * y) })
}
func Scalar(a *Tensor, v float32) *Tensor {
	b, _ := New([]float32{v}, []int{}, a.Device, false)
	b.ephemeral = true
	return b
}
func AddScalar(a *Tensor, v float32) *Tensor {
	if a.Device == tensor.CUDA {
		return gpuScalar(a, v, 0)
	}
	return Add(a, Scalar(a, v))
}
func SubScalar(a *Tensor, v float32) *Tensor {
	if a.Device == tensor.CUDA {
		return gpuScalar(a, v, 1)
	}
	return Sub(a, Scalar(a, v))
}
func MulScalar(a *Tensor, v float32) *Tensor {
	if a.Device == tensor.CUDA {
		return gpuScalar(a, v, 2)
	}
	return Mul(a, Scalar(a, v))
}
func DivScalar(a *Tensor, v float32) *Tensor {
	if a.Device == tensor.CUDA {
		return gpuScalar(a, v, 3)
	}
	return Div(a, Scalar(a, v))
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
	if a.Device == tensor.CUDA {
		return gpuUnary(a, 0)
	}
	return unary(a, func(x float32) float32 { return float32(math.Exp(float64(x))) }, func(x float32) float32 { return float32(math.Exp(float64(x))) })
}
func Log(a *Tensor) *Tensor {
	if a.Device == tensor.CUDA {
		return gpuUnary(a, 1)
	}
	return unary(a, func(x float32) float32 { return float32(math.Log(float64(x))) }, func(x float32) float32 { return 1 / x })
}
func Abs(a *Tensor) *Tensor {
	if a.Device == tensor.CUDA {
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
	if a.Device == tensor.CUDA {
		return gpuUnary(a, 3)
	}
	return unary(a, func(x float32) float32 { return float32(math.Tanh(float64(x))) }, func(x float32) float32 { v := float32(math.Tanh(float64(x))); return 1 - v*v })
}
func ReLU(a *Tensor) *Tensor {
	if a.Device == tensor.CUDA {
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
	if a.Device == tensor.CUDA {
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
