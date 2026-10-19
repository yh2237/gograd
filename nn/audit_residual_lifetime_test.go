package nn

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func auditLiveBytes(t *testing.T, device tensor.Device, want int) {
	t.Helper()
	if device == tensor.CUDA {
		if got := cuda.MemoryStats().LiveBytes; got != want {
			t.Fatalf("CUDA live bytes: got %d, want %d (leaked activation/workspace)", got, want)
		}
	}
}

func auditResidualLifetime(t *testing.T, device tensor.Device, blas *cuda.Blas) {
	for _, kind := range []string{"linear", "conv", "tanh", "sequential", "nested_residual"} {
		t.Run(kind, func(t *testing.T) {
			auditDeviceThread(t, device)
			start := cuda.MemoryStats().LiveBytes
			var inner Module
			rng := rand.New(rand.NewSource(57))
			switch kind {
			case "linear":
				m, err := NewLinearOn(device, 2, 2, rng)
				if err != nil {
					t.Fatal(err)
				}
				inner = m
			case "conv", "sequential":
				m, err := NewConv1dOn(device, 2, 2, 3, 1, rng)
				if err != nil {
					t.Fatal(err)
				}
				inner = m
				if kind == "sequential" {
					inner = &Sequential{Modules: []Module{m, &GELU{}, &Tanh{}}}
				}
			case "tanh":
				inner = &Tanh{}
			case "nested_residual":
				inner = &Residual{Inner: &Tanh{}}
			}
			r := &Residual{Inner: inner}
			x := tensorOn(t, device, []int{1, 3, 2}, []float32{-.7, .3, .4, -.2, .8, .1})
			g := tensorOn(t, device, []int{1, 3, 2}, []float32{.2, -.3, .5, .1, -.2, .4})
			baseline := cuda.MemoryStats().LiveBytes
			for iteration := 0; iteration < 3; iteration++ {
				// The caller owns the sum, but Residual must retire its old inner
				// output on repeated forwards even when backward was skipped.
				for repeat := 0; repeat < 2; repeat++ {
					first, err := r.Forward(blas, x)
					if err != nil {
						t.Fatal(err)
					}
					first.Close()
				}
				y, err := r.Forward(blas, x)
				if err != nil {
					t.Fatal(err)
				}
				if kind == "tanh" || kind == "sequential" {
					if r.innerOutput != nil {
						t.Fatal("Residual duplicated inner-owned output ownership")
					}
				} else if r.innerOutput == nil {
					t.Fatal("Residual lost caller-owned inner output")
				}
				dx, err := r.Backward(blas, g)
				if err != nil {
					t.Fatal(err)
				}
				if kind == "tanh" {
					xv, gv := host(t, x), host(t, g)
					want := make([]float32, len(xv))
					for i, v := range xv {
						z := float32(math.Tanh(float64(v)))
						want[i] = gv[i] * (2 - z*z)
					}
					auditCompare(t, "residual tanh dX", host(t, dx), want)
				}
				dx.Close()
				y.Close()
				r.CloseActivations()
				r.CloseActivations()
				auditLiveBytes(t, device, baseline)
			}
			// Normal enclosing Sequential cleanup, including a repeated forward.
			model := &Sequential{Modules: []Module{r, &Tanh{}}}
			for iteration := 0; iteration < 3; iteration++ {
				if _, err := model.Forward(blas, x); err != nil {
					t.Fatal(err)
				}
				if _, err := model.Forward(blas, x); err != nil {
					t.Fatal(err)
				}
				dx, err := model.Backward(blas, g)
				if err != nil {
					t.Fatal(err)
				}
				dx.Close()
				model.CloseActivations()
				auditLiveBytes(t, device, baseline)
			}
			// Close also cleans an outstanding forward, including conv columns.
			if _, err := model.Forward(blas, x); err != nil {
				t.Fatal(err)
			}
			model.Close()
			x.Close()
			g.Close()
			auditLiveBytes(t, device, start)
		})
	}
}

type auditFailingInner struct {
	ReLU
	mode string
	last *tensor.Tensor
}

func (m *auditFailingInner) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if m.mode == "shape" {
		shape = []int{2, 3} // same number of elements, different shape
	}
	y, err := tensor.NewOn(x.Device(), shape...)
	m.last = y
	if err == nil && m.mode == "forward_error" {
		err = fmt.Errorf("injected forward failure")
	}
	return y, err
}

func (m *auditFailingInner) Backward(_ *cuda.Blas, _ *tensor.Tensor) (*tensor.Tensor, error) {
	return nil, fmt.Errorf("injected backward failure")
}

type auditBorrowedOutput struct{ ReLU }

func (*auditBorrowedOutput) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	return x, nil
}

func auditResidualFailures(t *testing.T, device tensor.Device, blas *cuda.Blas) {
	x := tensorOn(t, device, []int{1, 3, 2}, []float32{1, 2, 3, 4, 5, 6})
	defer x.Close()
	g := tensorOn(t, device, []int{1, 3, 2}, []float32{1, 1, 1, 1, 1, 1})
	defer g.Close()
	baseline := cuda.MemoryStats().LiveBytes
	for _, mode := range []string{"shape", "forward_error", "backward_error"} {
		t.Run(mode, func(t *testing.T) {
			auditDeviceThread(t, device)
			inner := &auditFailingInner{mode: mode}
			r := &Residual{Inner: inner}
			y, err := r.Forward(blas, x)
			if mode == "backward_error" {
				if err != nil {
					t.Fatal(err)
				}
				y.Close()
				if _, err = r.Backward(blas, g); err == nil {
					t.Fatal("expected backward error")
				}
				r.CloseActivations()
			} else if err == nil {
				t.Fatal("expected forward error")
			}
			if r.innerOutput != nil {
				t.Fatal("failed pass retained inner output")
			}
			if device == tensor.CUDA && inner.last.Buffer() != nil {
				t.Fatal("failed pass leaked output buffer")
			}
			r.Close()
			auditLiveBytes(t, device, baseline)
		})
	}
	// A failed child must not strand an earlier activation in Sequential.
	for _, mode := range []string{"shape", "forward_error"} {
		inner := &Sequential{Modules: []Module{&Tanh{}, &auditFailingInner{mode: mode}}}
		r := &Residual{Inner: inner}
		if _, err := r.Forward(blas, x); err == nil {
			t.Fatal("expected nested forward error")
		}
		r.Close()
		auditLiveBytes(t, device, baseline)
	}
	// A real convolution failure also has an im2col workspace to release.
	for _, kind := range []string{"linear", "conv"} {
		var m Module
		var err error
		if kind == "linear" {
			m, err = NewLinearOn(device, 2, 3, rand.New(rand.NewSource(1)))
		} else {
			m, err = NewConv1dOn(device, 2, 3, 3, 1, rand.New(rand.NewSource(1)))
		}
		if err != nil {
			t.Fatal(err)
		}
		r := &Residual{Inner: m}
		if _, err = r.Forward(blas, x); err == nil {
			t.Fatal("expected unequal channel error")
		}
		r.Close()
		auditLiveBytes(t, device, baseline)
	}
	// Identity/borrowed outputs (including an empty Sequential) must not free x.
	for _, inner := range []Module{&auditBorrowedOutput{}, &Sequential{}} {
		r := &Residual{Inner: inner}
		y, err := r.Forward(blas, x)
		if err != nil {
			t.Fatal(err)
		}
		y.Close()
		r.Close()
		if device == tensor.CUDA && x.Buffer() == nil {
			t.Fatal("cleanup freed borrowed input")
		}
		auditCompare(t, "borrowed input", host(t, x), []float32{1, 2, 3, 4, 5, 6})
		auditLiveBytes(t, device, baseline)
	}
}

func TestAuditResidualLifetimeCPU(t *testing.T)  { auditResidualLifetime(t, tensor.CPU, nil) }
func TestAuditResidualLifetimeCUDA(t *testing.T) { auditResidualLifetime(t, tensor.CUDA, auditCUDA(t)) }
func TestAuditResidualFailuresCPU(t *testing.T)  { auditResidualFailures(t, tensor.CPU, nil) }
func TestAuditResidualFailuresCUDA(t *testing.T) { auditResidualFailures(t, tensor.CUDA, auditCUDA(t)) }
