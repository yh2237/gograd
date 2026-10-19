package nn

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func auditCUDA(t *testing.T) *cuda.Blas {
	t.Helper()
	if !cuda.Available() {
		t.Skip("CUDA unavailable")
	}
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	blas, err := cuda.NewBlas()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { blas.Destroy() })
	return blas
}

func auditDeviceThread(t *testing.T, device tensor.Device) {
	t.Helper()
	if device == tensor.CUDA {
		runtime.LockOSThread()
		t.Cleanup(runtime.UnlockOSThread)
		if err := cuda.SetDevice(0); err != nil {
			t.Fatal(err)
		}
	}
}

func auditCompare(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", label, len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 3e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d]: got %g, want %g", label, i, v, want[i])
		}
	}
}

func auditMicrobatches(t *testing.T, device tensor.Device, blas *cuda.Blas) {
	for _, dims := range [][3]int{{1, 2, 3}, {3, 7, 5}, {11, 13, 7}, {273, 17, 9}} {
		rows, in, out := dims[0], dims[1], dims[2]
		t.Run(fmt.Sprintf("linear_%dx%dx%d", rows, in, out), func(t *testing.T) {
			auditDeviceThread(t, device)
			m, err := NewLinearOn(device, in, out, rand.New(rand.NewSource(42)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			auditAccumulate(t, device, blas, m, []int{rows, in}, []int{rows, out}, in, out, 1, 1)
		})
	}
	t.Run("conv_dilated", func(t *testing.T) {
		auditDeviceThread(t, device)
		m, err := NewConv1dOn(device, 2, 3, 3, 2, rand.New(rand.NewSource(42)))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		auditAccumulate(t, device, blas, m, []int{2, 5, 2}, []int{2, 5, 3}, 2, 3, 3, 2)
	})
}

func auditAccumulate(t *testing.T, device tensor.Device, blas *cuda.Blas, m Module, xs, gs []int, in, out, kernel, dilation int) {
	t.Helper()
	wantW := make([]float32, out*in*kernel)
	wantB := make([]float32, out)
	w := host(t, m.Parameters()[0])
	rows := tensor.Numel(xs) / in
	length := rows
	if len(xs) == 3 {
		length = xs[1]
	}
	m.ZeroGrad()
	var scratch *float32
	for micro := 0; micro < 3; micro++ {
		if micro == 2 {
			m.ZeroGrad()
			clear(wantW)
			clear(wantB)
		}
		xv, gv := make([]float32, tensor.Numel(xs)), make([]float32, tensor.Numel(gs))
		for i := range xv {
			xv[i] = float32(math.Sin(float64(i+micro*7) * .31))
		}
		for i := range gv {
			gv[i] = float32(math.Cos(float64(i+micro*11) * .23))
		}
		wantX := make([]float32, len(xv))
		for row := 0; row < rows; row++ {
			for o := 0; o < out; o++ {
				g := gv[row*out+o]
				wantB[o] += g
				for ci := 0; ci < in; ci++ {
					for k := 0; k < kernel; k++ {
						at := row%length + (k-kernel/2)*dilation
						if at < 0 || at >= length {
							continue
						}
						xi := ((row/length)*length+at)*in + ci
						wi := (o*in+ci)*kernel + k
						wantW[wi] += g * xv[xi]
						wantX[xi] += g * w[wi]
					}
				}
			}
		}
		x := tensorOn(t, device, xs, xv)
		g := tensorOn(t, device, gs, gv)
		y, err := m.Forward(blas, x)
		if err != nil {
			t.Fatal(err)
		}
		dx, err := m.Backward(blas, g)
		if err != nil {
			t.Fatal(err)
		}
		if l, ok := m.(*Linear); ok && len(l.weightGradScratch) > 0 {
			if scratch != nil && scratch != &l.weightGradScratch[0] {
				t.Fatal("CPU weight gradient workspace was reallocated")
			}
			scratch = &l.weightGradScratch[0]
		}
		auditCompare(t, "dX", host(t, dx), wantX)
		auditCompare(t, fmt.Sprintf("micro %d dW", micro), host(t, m.Gradients()[0]), wantW)
		auditCompare(t, fmt.Sprintf("micro %d db", micro), host(t, m.Gradients()[1]), wantB)
		dx.Close()
		y.Close()
		g.Close()
		x.Close()
	}
}

func TestAuditMicrobatchCPU(t *testing.T) { auditMicrobatches(t, tensor.CPU, nil) }

func TestAuditMicrobatchCUDA(t *testing.T) {
	auditMicrobatches(t, tensor.CUDA, auditCUDA(t))
}

// Measures the complete CPU backward (dX, accumulated dW and db), not just GEMM.
func BenchmarkAuditLinearBackwardCPU(b *testing.B) {
	auditBenchmarkLinearBackward(b, false)
}

// The former serial implementation is retained only as a performance baseline.
func BenchmarkAuditLinearBackwardSerialCPU(b *testing.B) {
	auditBenchmarkLinearBackward(b, true)
}

func auditBenchmarkLinearBackward(b *testing.B, serial bool) {
	for _, dims := range [][3]int{{1, 4, 4}, {32, 64, 64}, {256, 256, 256}, {1024, 256, 128}} {
		rows, in, out := dims[0], dims[1], dims[2]
		b.Run(fmt.Sprintf("%dx%dx%d", rows, in, out), func(b *testing.B) {
			m, err := NewLinearOn(tensor.CPU, in, out, rand.New(rand.NewSource(42)))
			if err != nil {
				b.Fatal(err)
			}
			defer m.Close()
			x, _ := tensor.ZerosOn(tensor.CPU, rows, in)
			g, _ := tensor.ZerosOn(tensor.CPU, rows, out)
			defer x.Close()
			defer g.Close()
			for i := range x.Data() {
				x.Data()[i] = float32(i%17-8) * .01
			}
			for i := range g.Data() {
				g.Data()[i] = float32(i%11-5) * .01
			}
			y, err := m.Forward(nil, x)
			if err != nil {
				b.Fatal(err)
			}
			defer y.Close()
			backward := m.Backward
			if serial {
				backward = func(_ *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error) {
					dx, err := tensor.NewOn(tensor.CPU, m.input.Shape()...)
					if err != nil {
						return nil, err
					}
					tensor.SGEMM(dx.Data(), grad.Data(), m.Weight.Data(), m.rows, m.in, m.out)
					for r := 0; r < m.rows; r++ {
						for o := 0; o < m.out; o++ {
							g := grad.Data()[r*m.out+o]
							m.gradBias.Data()[o] += g
							for i := 0; i < m.in; i++ {
								m.gradWeight.Data()[o*m.in+i] += g * m.input.Data()[r*m.in+i]
							}
						}
					}
					return dx, nil
				}
			}
			// Warm the reusable dW workspace and SGEMM packing pools.
			dx, err := backward(nil, g)
			if err != nil {
				b.Fatal(err)
			}
			dx.Close()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dx, err := backward(nil, g)
				if err != nil {
					b.Fatal(err)
				}
				dx.Close()
			}
		})
	}
}
