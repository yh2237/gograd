// Package autograd implements a define-by-run float32 tensor graph. The CUDA
// device currently uses host staging for operators without a device kernel.
package autograd

import (
	"errors"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
)

type Tensor struct {
	Data, Grad     []float32
	Shape, Strides []int
	DType          DType
	Device         tensor.Device
	RequiresGrad   bool
	parents        []*Tensor
	backward       func([]float32)
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
	return &Tensor{Data: append([]float32(nil), data...), Shape: append([]int(nil), shape...), Strides: strides(shape), DType: Float32, Device: device, RequiresGrad: grad}, nil
}
func Must(data []float32, shape []int, grad bool) *Tensor {
	v, e := New(data, shape, tensor.CPU, grad)
	if e != nil {
		panic(e)
	}
	return v
}
func Zeros(shape []int, device tensor.Device, grad bool) (*Tensor, error) {
	return New(make([]float32, numel(shape)), shape, device, grad)
}
func (t *Tensor) Detach() *Tensor {
	v := *t
	v.Data = append([]float32(nil), t.Data...)
	v.Grad = nil
	v.RequiresGrad = false
	v.parents = nil
	v.backward = nil
	return &v
}
func (t *Tensor) RetainGrad() { t.retain = true }
func (t *Tensor) ZeroGrad()   { t.Grad = nil }
func result(data []float32, shape []int, parents []*Tensor, back func([]float32)) *Tensor {
	req := false
	for _, p := range parents {
		req = req || p.RequiresGrad
	}
	return &Tensor{Data: data, Shape: shape, Strides: strides(shape), DType: Float32, Device: parents[0].Device, RequiresGrad: req && recording, parents: parents, backward: back}
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
	if len(t.Data) != 1 {
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
	t.addGrad([]float32{1})
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
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
func binary(a, b *Tensor, f, da, db func(float32, float32) float32) *Tensor {
	same(a, b)
	s := bshape(a.Shape, b.Shape)
	v := make([]float32, numel(s))
	for i := range v {
		v[i] = f(a.Data[bindex(i, s, a.Shape)], b.Data[bindex(i, s, b.Shape)])
	}
	return result(v, s, []*Tensor{a, b}, func(g []float32) {
		ag := make([]float32, len(a.Data))
		bg := make([]float32, len(b.Data))
		for i, z := range g {
			ai, bi := bindex(i, s, a.Shape), bindex(i, s, b.Shape)
			x, y := a.Data[ai], b.Data[bi]
			ag[ai] += z * da(x, y)
			bg[bi] += z * db(x, y)
		}
		a.addGrad(ag)
		b.addGrad(bg)
	})
}
func Add(a, b *Tensor) *Tensor {
	return binary(a, b, func(x, y float32) float32 { return x + y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return 1 })
}
func Sub(a, b *Tensor) *Tensor {
	return binary(a, b, func(x, y float32) float32 { return x - y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return -1 })
}
func Mul(a, b *Tensor) *Tensor {
	return binary(a, b, func(x, y float32) float32 { return x * y }, func(x, y float32) float32 { return y }, func(x, y float32) float32 { return x })
}
func Div(a, b *Tensor) *Tensor {
	return binary(a, b, func(x, y float32) float32 { return x / y }, func(x, y float32) float32 { return 1 / y }, func(x, y float32) float32 { return -x / (y * y) })
}
func Scalar(a *Tensor, v float32) *Tensor {
	b, _ := New([]float32{v}, []int{}, a.Device, false)
	return b
}
func AddScalar(a *Tensor, v float32) *Tensor { return Add(a, Scalar(a, v)) }
func SubScalar(a *Tensor, v float32) *Tensor { return Sub(a, Scalar(a, v)) }
func MulScalar(a *Tensor, v float32) *Tensor { return Mul(a, Scalar(a, v)) }
func DivScalar(a *Tensor, v float32) *Tensor { return Div(a, Scalar(a, v)) }
func unary(a *Tensor, f, d func(float32) float32) *Tensor {
	v := make([]float32, len(a.Data))
	for i, x := range a.Data {
		v[i] = f(x)
	}
	return result(v, append([]int(nil), a.Shape...), []*Tensor{a}, func(g []float32) {
		dx := make([]float32, len(g))
		for i, x := range a.Data {
			dx[i] = g[i] * d(x)
		}
		a.addGrad(dx)
	})
}
func Exp(a *Tensor) *Tensor {
	return unary(a, func(x float32) float32 { return float32(math.Exp(float64(x))) }, func(x float32) float32 { return float32(math.Exp(float64(x))) })
}
func Log(a *Tensor) *Tensor {
	return unary(a, func(x float32) float32 { return float32(math.Log(float64(x))) }, func(x float32) float32 { return 1 / x })
}
func Abs(a *Tensor) *Tensor {
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
	return unary(a, func(x float32) float32 { return float32(math.Tanh(float64(x))) }, func(x float32) float32 { v := float32(math.Tanh(float64(x))); return 1 - v*v })
}
func ReLU(a *Tensor) *Tensor {
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
