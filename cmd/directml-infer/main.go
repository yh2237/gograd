// directml-infer runs resident GEMM and pointwise inference with explicit CPU
// fallback chosen before tensor construction. It is not a performance benchmark.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/directml"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	m := flag.Int("m", 74, "output rows (1..2048)")
	n := flag.Int("n", 768, "output columns (1..2048)")
	k := flag.Int("k", 768, "reduction dimension (1..2048)")
	fallback := flag.Bool("cpu-fallback", false, "choose CPU before construction if DirectML is unavailable")
	encoder := flag.Bool("encoder", false, "run a small batch-first TransformerEncoderLayer and compare with CPU")
	flag.Parse()
	for _, dim := range []int{*m, *n, *k} {
		if dim < 1 || dim > 2048 {
			return fmt.Errorf("dimensions must be in 1..2048")
		}
	}
	device := tensor.DirectML
	if !directml.Available() {
		if !*fallback {
			return directml.Err()
		}
		device = tensor.CPU
	}
	if device == tensor.DirectML {
		info, err := directml.DeviceInfo()
		if err != nil {
			return err
		}
		fmt.Printf("DirectML GPU: %s (feature level 0x%x)\n", info.Name, info.FeatureLevel)
		if *encoder && info.FeatureLevel < 0x5100 {
			return fmt.Errorf("%w: encoder requires native GELU (DirectML feature level 5.1)", gpu.ErrUnsupported)
		}
	}
	if *encoder {
		return inferEncoder(device)
	}
	return infer(device, *m, *n, *k)
}

func inferEncoder(device tensor.Device) error {
	c, reference := autograd.NewExecutionContext(), autograd.NewExecutionContext()
	layer, err := autograd.NewTransformerEncoderLayer(16, 4, 48, device)
	if err != nil {
		return err
	}
	for _, p := range layer.NamedParameters() {
		defer p.Value.Close()
	}
	cpu, err := autograd.NewTransformerEncoderLayer(16, 4, 48, tensor.CPU)
	if err != nil {
		return err
	}
	for _, p := range cpu.NamedParameters() {
		defer p.Value.Close()
	}
	layer.Training, layer.Attention.Training = false, false
	cpu.Training, cpu.Attention.Training = false, false
	if err := c.BindModule(&layer.Module); err != nil {
		return err
	}
	if err := reference.BindModule(&cpu.Module); err != nil {
		return err
	}
	data := make([]float32, 2*17*16)
	for i := range data {
		data[i] = float32(i%97-48) / 32
	}
	x, err := c.New(data, []int{2, 17, 16}, device, true)
	if err != nil {
		return err
	}
	defer x.Close()
	cx, err := reference.New(data, []int{2, 17, 16}, tensor.CPU, true)
	if err != nil {
		return err
	}
	defer cx.Close()
	var y, ref *autograd.Tensor
	c.NoGrad(func() { y = layer.Forward(x, nil, 0) })
	defer y.ReleaseGraph()
	reference.NoGrad(func() { ref = cpu.Forward(cx, nil, 0) })
	defer ref.ReleaseGraph()
	values, err := y.ToHost()
	if err != nil {
		return err
	}
	var maxError, dot, xx, yy float64
	for i, v := range values {
		want := ref.Data[i]
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite encoder output at %d", i)
		}
		diff := math.Abs(float64(v - want))
		maxError = max(maxError, diff)
		if diff > 1e-4+1e-4*math.Abs(float64(want)) {
			return fmt.Errorf("encoder CPU agreement failed at %d", i)
		}
		dot += float64(v) * float64(want)
		xx += float64(v) * float64(v)
		yy += float64(want) * float64(want)
	}
	cosine := dot / math.Sqrt(xx*yy)
	fmt.Printf("device=%s encoder=[2,17,16] heads=4 ffn=48 max_abs_error=%.9g cosine=%.12f\n", device, maxError, cosine)
	if math.IsNaN(cosine) || cosine < .999 {
		return fmt.Errorf("encoder cosine agreement failed")
	}
	return nil
}

func infer(device tensor.Device, m, n, k int) error {
	a, w := make([]float32, m*k), make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%37-18) / 32
	}
	for i := range w {
		w[i] = float32(i%13-6) / 16
	}
	c := autograd.NewExecutionContext()
	x, err := c.New(a, []int{m, k}, device, true)
	if err != nil {
		return err
	}
	defer x.Close()
	weight, err := c.New(w, []int{k, n}, device, true)
	if err != nil {
		return err
	}
	defer weight.Close()
	var out *autograd.Tensor
	c.NoGrad(func() {
		product := autograd.MatMul(x, weight)
		out = autograd.MulScalar(autograd.Add(product, product), .25)
	})
	defer out.ReleaseGraph()
	values, err := out.ToHost()
	if err != nil {
		return err
	}
	want := make([]float32, m*n)
	tensor.SGEMM(want, a, w, m, n, k)
	var maxError float64
	for i, v := range values {
		want[i] *= .5
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite output at %d", i)
		}
		maxError = max(maxError, math.Abs(float64(v-want[i])))
	}
	fmt.Printf("device=%s GEMM=[%d,%d] @ [%d,%d] max_abs_error=%.9g\n", device, m, k, k, n, maxError)
	if maxError > 1e-5 {
		return fmt.Errorf("CPU agreement failed")
	}
	return nil
}
