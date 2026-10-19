package nn

import (
	"math/rand"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// gradProbe behaves like an identity layer: it returns the gradient it
// received. It also records whether that gradient was still usable, which
// catches a Sequential that releases a gradient another module still reads.
type gradProbe struct {
	seen  *tensor.Tensor
	alive bool
}

func (p *gradProbe) Parameters() []*tensor.Tensor { return nil }
func (p *gradProbe) Gradients() []*tensor.Tensor  { return nil }
func (p *gradProbe) ZeroGrad()                    {}
func (p *gradProbe) Forward(_ *cuda.Blas, x *tensor.Tensor) (*tensor.Tensor, error) {
	return x, nil
}
func (p *gradProbe) Backward(_ *cuda.Blas, grad *tensor.Tensor) (*tensor.Tensor, error) {
	p.seen = grad
	p.alive = grad.Device() != tensor.CUDA || grad.Buffer() != nil
	return grad, nil
}

func auditAliasedLinear(t *testing.T, device tensor.Device, in, out int, weight []float32) *Linear {
	t.Helper()
	linear, err := NewLinearOn(device, in, out, rand.New(rand.NewSource(3)))
	if err != nil {
		t.Fatal(err)
	}
	if err := linear.Weight.CopyFrom(weight); err != nil {
		linear.Close()
		t.Fatal(err)
	}
	if err := linear.Bias.Zero(); err != nil {
		linear.Close()
		t.Fatal(err)
	}
	return linear
}

// TestAuditSequentialBackwardKeepsAliasedGradient covers a chain that
// contains an identity module. Sequential must not release the gradient that
// the identity handed back, because the next module still reads it.
func TestAuditSequentialBackwardKeepsAliasedGradient(t *testing.T) {
	t.Run("cpu", func(t *testing.T) {
		auditSequentialAliasedGradient(t, tensor.CPU, nil)
	})
	t.Run("cuda", func(t *testing.T) {
		blas := auditCUDA(t)
		auditSequentialAliasedGradient(t, tensor.CUDA, blas)
	})
}

func auditSequentialAliasedGradient(t *testing.T, device tensor.Device, blas *cuda.Blas) {
	t.Helper()
	first := auditAliasedLinear(t, device, 2, 2, []float32{1, 0, 0, 1})
	defer first.Close()
	probe := &gradProbe{}
	second := auditAliasedLinear(t, device, 2, 1, []float32{1, 1})
	defer second.Close()
	seq := &Sequential{Modules: []Module{first, probe, second}}
	x, err := tensor.FromHostOn(device, []int{1, 2}, []float32{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	out, err := seq.Forward(blas, x)
	if err != nil {
		t.Fatal(err)
	}
	values, err := out.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	auditCompare(t, "forward", values, []float32{5})
	grad, err := tensor.FromHostOn(device, []int{1, 1}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer grad.Close()
	dx, err := seq.Backward(blas, grad)
	if err != nil {
		t.Fatal(err)
	}
	defer dx.Close()
	if !probe.alive {
		t.Fatal("sequential backward released a gradient that is still in use")
	}
	if probe.seen == nil {
		t.Fatal("identity module did not receive a gradient")
	}
	gradient, err := dx.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	auditCompare(t, "gradient", gradient, []float32{1, 1})
}
