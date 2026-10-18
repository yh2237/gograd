// vulkan-infer demonstrates a device-resident M1 inference chain. CPU fallback
// is explicit and selects the device before any tensors are constructed.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
	"github.com/yh2237/gograd/vulkan"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	n := flag.Int("n", 1024, "number of float32 elements (1..16777216)")
	fallback := flag.Bool("cpu-fallback", false, "choose CPU before construction when Vulkan is unavailable")
	linear := flag.Bool("linear", false, "run a batched 74-to-768 linear projection (batch=2, time=768)")
	flag.Parse()
	if *n < 1 || *n > 1<<24 {
		return fmt.Errorf("invalid element count")
	}
	device := tensor.Vulkan
	if !vulkan.Available() {
		if !*fallback {
			return vulkan.Err()
		}
		device = tensor.CPU
	}
	if device == tensor.Vulkan {
		info, err := vulkan.DeviceInfo()
		if err != nil {
			return err
		}
		fmt.Printf("Vulkan GPU: %s (API %d.%d.%d, compute queue %d)\n", info.Name, info.APIVersion>>22, info.APIVersion>>12&1023, info.APIVersion&4095, info.QueueFamily)
	}
	if *linear {
		return runLinear(device)
	}
	a, b := make([]float32, *n), make([]float32, *n)
	for i := range a {
		a[i] = float32(i%97-48) / 16
		b[i] = float32(i%37-18) / 32
	}
	execution := autograd.NewExecutionContext()
	x, err := execution.New(a, []int{*n}, device, true)
	if err != nil {
		return err
	}
	defer x.Close()
	w, err := execution.New(b, []int{*n}, device, true)
	if err != nil {
		return err
	}
	defer w.Close()
	var y *autograd.Tensor
	execution.NoGrad(func() { y = autograd.MulScalar(autograd.Add(autograd.Mul(x, w), x), .25) })
	defer y.ReleaseGraph()
	values, err := y.ToHost()
	if err != nil {
		return err
	}
	maxError := float64(0)
	for i, value := range values {
		want := (a[i]*b[i] + a[i]) * .25
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite output at %d", i)
		}
		maxError = max(maxError, math.Abs(float64(value-want)))
	}
	fmt.Printf("device=%s elements=%d max_abs_error=%.9g\n", device, *n, maxError)
	if maxError > 1e-5 {
		return fmt.Errorf("CPU agreement failed")
	}
	return nil
}

func runLinear(device tensor.Device) error {
	c, reference := autograd.NewExecutionContext(), autograd.NewExecutionContext()
	layer, err := autograd.NewLinearLayer(74, 768, device, rand.New(rand.NewSource(17)))
	if err != nil {
		return err
	}
	defer layer.Weight.Close()
	defer layer.Bias.Close()
	cpu, err := autograd.NewLinearLayer(74, 768, tensor.CPU, rand.New(rand.NewSource(17)))
	if err != nil {
		return err
	}
	defer cpu.Weight.Close()
	defer cpu.Bias.Close()
	if err := c.BindModule(&layer.Module); err != nil {
		return err
	}
	if err := reference.BindModule(&cpu.Module); err != nil {
		return err
	}
	data := make([]float32, 2*768*74)
	for i := range data {
		data[i] = float32(i%97-48) / 32
	}
	x, err := c.New(data, []int{2, 768, 74}, device, true)
	if err != nil {
		return err
	}
	defer x.Close()
	cx, err := reference.New(data, []int{2, 768, 74}, tensor.CPU, true)
	if err != nil {
		return err
	}
	defer cx.Close()
	var y, ref *autograd.Tensor
	c.NoGrad(func() { y = layer.Forward(x) })
	defer y.ReleaseGraph()
	reference.NoGrad(func() { ref = cpu.Forward(cx) })
	defer ref.ReleaseGraph()
	values, err := y.ToHost()
	if err != nil {
		return err
	}
	maxError := float64(0)
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite output at %d", i)
		}
		diff := math.Abs(float64(value - ref.Data[i]))
		maxError = max(maxError, diff)
		if diff > 1e-4+1e-4*math.Abs(float64(ref.Data[i])) {
			return fmt.Errorf("linear CPU agreement failed at %d", i)
		}
	}
	fmt.Printf("device=%s linear=[2,768,74] @ [74,768] + bias max_abs_error=%.9g\n", device, maxError)
	return nil
}
