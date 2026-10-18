package autograd

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func TestCPUUnaryParallel(t *testing.T) {
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)
	for _, tc := range []struct {
		name string
		op   func(*Tensor) *Tensor
	}{
		{"exp", Exp}, {"log", Log}, {"abs", Abs}, {"tanh", Tanh},
		{"sin", Sin}, {"relu", ReLU},
		{"gelu", func(x *Tensor) *Tensor { return GELU(x, false) }},
		{"gelu_approx", func(x *Tensor) *Tensor { return GELU(x, true) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x, _ := Zeros([]int{4, 256, 256}, tensor.CPU, true)
			defer x.Close()
			g := make([]float32, x.Numel())
			for i := range x.Data {
				x.Data[i] = float32(i%97-48) / 16
				if tc.name == "log" {
					x.Data[i] += 4
				}
				g[i] = float32(i%13-6) / 8
			}
			runtime.GOMAXPROCS(1)
			serial := tc.op(x)
			serial.backward(g)
			want, grad := append([]float32(nil), serial.Data...), append([]float32(nil), x.Grad...)
			serial.ReleaseGraph()
			x.ZeroGrad()
			runtime.GOMAXPROCS(4)
			parallel := tc.op(x)
			defer parallel.ReleaseGraph()
			parallel.backward(g)
			conv2dClose(t, "forward", parallel.Data, want, 0, 0)
			conv2dClose(t, "VJP", x.Grad, grad, 0, 0)
		})
	}
}

func TestCPUSoftmaxReference(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	for _, tc := range []struct {
		shape []int
		axis  int
	}{
		{[]int{13}, -1},
		{[]int{3, 1, 7}, 1},
		{[]int{2, 5, 7}, 0},
		{[]int{2, 3, 5, 7}, 2},
		{[]int{16, 4, 128, 64}, -1},
		{[]int{16, 256, 64}, 1},
		{[]int{0, 7}, -1},
	} {
		for _, logmode := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/axis%d/log%t", tc.shape, tc.axis, logmode), func(t *testing.T) {
				data, g := make([]float32, numel(tc.shape)), make([]float32, numel(tc.shape))
				for i := range data {
					data[i] = float32(i%97-48) / 8
					g[i] = float32(i%13-6) / 16
				}
				if len(data) > 1 {
					data[0], data[1] = -1000, 1000
				}
				x := conv2dTestTensor(t, data, tc.shape, tensor.CPU, true)
				var y *Tensor
				if logmode {
					y = LogSoftmax(x, tc.axis)
				} else {
					y = Softmax(x, tc.axis)
				}
				defer y.ReleaseGraph()
				want, grad := referenceCPUSoftmax(data, g, tc.shape, tc.axis, logmode)
				// Reusing exp and splitting independent rows must not change any
				// rounding, including on a non-final axis.
				for i, v := range want {
					if math.Float32bits(y.Data[i]) != math.Float32bits(v) {
						t.Fatalf("output[%d]=%g want %g", i, y.Data[i], v)
					}
				}
				x.Grad = make([]float32, x.Numel())
				for i := range x.Grad {
					x.Grad[i] = .25
					grad[i] += .25
				}
				y.backward(g)
				conv2dClose(t, "VJP", x.Grad, grad, 0, 0)
			})
		}
	}
}

// The original scalar formulation is deliberately independent of the optimized
// row scheduling and temporary exp reuse.
func referenceCPUSoftmax(x, g []float32, shape []int, axis int, logmode bool) ([]float32, []float32) {
	y, dx := make([]float32, len(x)), make([]float32, len(x))
	outer, dim, inner := axisParts(shape, axis)
	for row := 0; row < outer*inner; row++ {
		base := row/inner*dim*inner + row%inner
		mx := float32(math.Inf(-1))
		for k := 0; k < dim; k++ {
			mx = max(mx, x[base+k*inner])
		}
		var sum float32
		for k := 0; k < dim; k++ {
			sum += float32(math.Exp(float64(x[base+k*inner] - mx)))
		}
		ls := float32(math.Log(float64(sum)))
		for k := 0; k < dim; k++ {
			i := base + k*inner
			if logmode {
				y[i] = x[i] - mx - ls
			} else {
				y[i] = float32(math.Exp(float64(x[i]-mx))) / sum
			}
		}
		var dot float32
		for k := 0; k < dim; k++ {
			i := base + k*inner
			if logmode {
				dot += g[i]
			} else {
				dot += g[i] * y[i]
			}
		}
		for k := 0; k < dim; k++ {
			i := base + k*inner
			if logmode {
				dx[i] = g[i] - float32(math.Exp(float64(y[i])))*dot
			} else {
				dx[i] = y[i] * (g[i] - dot)
			}
		}
	}
	return y, dx
}

func TestCPUMatMulGradientSelection(t *testing.T) {
	for _, tc := range []struct{ a, b []int }{
		{[]int{37, 19}, []int{19, 11}},
		{[]int{1, 1, 7, 5}, []int{5, 9}},
		{[]int{2, 7, 5}, []int{5, 9}},
		{[]int{2, 7, 5}, []int{2, 5, 9}},
		{[]int{2, 65, 17}, []int{2, 17, 19}},
		{[]int{7, 5}, []int{3, 5, 9}},
		{[]int{2, 1, 7, 5}, []int{1, 3, 5, 9}},
		{[]int{5, 0}, []int{0, 7}},
	} {
		for _, requires := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
			t.Run(fmt.Sprintf("%v_%v/grad%v", tc.a, tc.b, requires), func(t *testing.T) {
				av, bv := make([]float32, numel(tc.a)), make([]float32, numel(tc.b))
				for i := range av {
					av[i] = float32(i%17-8) / 16
				}
				for i := range bv {
					bv[i] = float32(i%13-6) / 16
				}
				a := conv2dTestTensor(t, av, tc.a, tensor.CPU, requires[0])
				b := conv2dTestTensor(t, bv, tc.b, tensor.CPU, requires[1])
				y := MatMul(a, b)
				defer y.ReleaseGraph()
				m, k, n := tc.a[len(tc.a)-2], tc.a[len(tc.a)-1], tc.b[len(tc.b)-1]
				batch := y.Shape[:len(y.Shape)-2]
				want, da, db, up := make([]float32, y.Numel()), make([]float32, len(av)), make([]float32, len(bv)), make([]float32, y.Numel())
				for i := range up {
					up[i] = float32(i%7-3) / 8
				}
				for z := 0; z < numel(batch); z++ {
					ai := bindex(z, batch, tc.a[:len(tc.a)-2]) * m * k
					bi := bindex(z, batch, tc.b[:len(tc.b)-2]) * k * n
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							o := (z*m+i)*n + j
							for d := 0; d < k; d++ {
								ia, ib := ai+i*k+d, bi+d*n+j
								want[o] += av[ia] * bv[ib]
								da[ia] += up[o] * bv[ib]
								db[ib] += av[ia] * up[o]
							}
						}
					}
				}
				conv2dClose(t, "forward", y.Data, want, 1e-5, 1e-5)
				if !requires[0] && !requires[1] {
					return
				}
				g := conv2dTestTensor(t, up, y.Shape, tensor.CPU, false)
				axes := make([]int, len(y.Shape))
				for i := range axes {
					axes[i] = i
				}
				loss := Sum(Mul(y, g), axes...)
				defer loss.ReleaseGraph()
				for pass := 0; pass < 2; pass++ {
					if err := loss.BackwardWithOptions(BackwardOptions{RetainGraph: pass == 0}); err != nil {
						t.Fatal(err)
					}
				}
				for i := range da {
					da[i] *= 2
				}
				for i := range db {
					db[i] *= 2
				}
				if requires[0] {
					conv2dClose(t, "grad A", a.Grad, da, 1e-5, 1e-5)
				} else if a.Grad != nil {
					t.Fatal("gradient allocated for frozen A")
				}
				if requires[1] {
					conv2dClose(t, "grad B", b.Grad, db, 1e-5, 1e-5)
				} else if b.Grad != nil {
					t.Fatal("gradient allocated for frozen B")
				}
			})
		}
	}
}

func TestCPUMatMulSharedOperand(t *testing.T) {
	x := conv2dTestTensor(t, []float32{1, 2, 3, 4}, []int{2, 2}, tensor.CPU, true)
	loss := Sum(MatMul(x, x), 0, 1)
	defer loss.ReleaseGraph()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	conv2dClose(t, "shared operand VJP", x.Grad, []float32{7, 11, 9, 13}, 0, 0)
}
