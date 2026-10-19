// gpu-infer-bench measures synchronized, resident FP32 inference through the
// same autograd API on CPU, Vulkan and DirectML. It never substitutes a device.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/directml"
	"github.com/yh2237/gograd/gpu"
	"github.com/yh2237/gograd/tensor"
	"github.com/yh2237/gograd/vulkan"
)

type config struct {
	Devices, Workloads                                     string
	Size, Hidden, Heads, Layers, Batch, Sequence, Channels int
	Warmup, Iterations                                     int
	SkipUnavailable, JSON                                  bool
}
type memory struct {
	LiveBytes, CachedBytes, ReservedBytes uint64
	LiveBuffers                           int
	Operators                             int
	OperatorBytes                         uint64
}
type result struct {
	Workload       string        `json:"workload"`
	Device         tensor.Device `json:"device"`
	SetupMS        float64       `json:"setup_ms"`
	FirstForwardMS float64       `json:"first_forward_ms"`
	MedianMS       float64       `json:"median_ms"`
	P95MS          float64       `json:"p95_ms"`
	ReadbackMS     float64       `json:"readback_ms"`
	MaxAbsError    float64       `json:"max_abs_error"`
	Cosine         float64       `json:"cosine"`
	Speedup        float64       `json:"speedup_vs_cpu,omitempty"`
	SamplesMS      []float64     `json:"samples_ms"`
	Memory         *memory       `json:"memory_after_release,omitempty"`
}
type deviceInfo struct {
	Device      tensor.Device `json:"device"`
	Name        string        `json:"name"`
	InitMS      float64       `json:"init_ms"`
	Unavailable string        `json:"unavailable,omitempty"`
}
type report struct {
	GoVersion, OS, Arch string
	CPUs, GOMAXPROCS    int
	Config              config
	Devices             []deviceInfo
	Results             []result
}
type workload struct {
	forward func() *autograd.Tensor
	leaves  []*autograd.Tensor
}

func (w *workload) close() {
	for _, v := range w.leaves {
		v.Close()
	}
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(args []string, out io.Writer) error {
	c := config{}
	fs := flag.NewFlagSet("gpu-infer-bench", flag.ContinueOnError)
	fs.StringVar(&c.Devices, "devices", "cpu,vulkan,directml", "comma-separated devices; no automatic CPU fallback")
	fs.StringVar(&c.Workloads, "workloads", "gemm,projection,ffn,encoder", "gemm, projection, ffn, encoder, gelu (tanh approximation), layernorm, bias_gelu (erf), bias_residual")
	fs.IntVar(&c.Size, "size", 768, "model width and square GEMM dimension")
	fs.IntVar(&c.Hidden, "hidden", 3072, "feed-forward width")
	fs.IntVar(&c.Heads, "heads", 12, "encoder attention heads")
	fs.IntVar(&c.Layers, "layers", 12, "encoder layers after Conv1d")
	fs.IntVar(&c.Batch, "batch", 1, "batch size")
	fs.IntVar(&c.Sequence, "seq", 74, "sequence length")
	fs.IntVar(&c.Channels, "channels", 74, "Conv1d input channels")
	fs.IntVar(&c.Warmup, "warmup", 2, "untimed forwards after the separately reported first forward")
	fs.IntVar(&c.Iterations, "iters", 5, "measured forwards")
	fs.BoolVar(&c.SkipUnavailable, "skip-unavailable", true, "report missing devices and continue; never replace them")
	fs.BoolVar(&c.JSON, "json", false, "emit environment, individual samples and results as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
	}
	if err := validate(c); err != nil {
		return err
	}
	r := report{GoVersion: runtime.Version(), OS: runtime.GOOS, Arch: runtime.GOARCH, CPUs: runtime.NumCPU(), GOMAXPROCS: runtime.GOMAXPROCS(0), Config: c}
	var devices []tensor.Device
	for _, name := range strings.Split(c.Devices, ",") {
		device := tensor.Device(strings.TrimSpace(name))
		start := clockNow()
		info := deviceInfo{Device: device, Name: string(device)}
		if device != tensor.CPU {
			b, err := gpu.Get(string(device))
			if err != nil {
				if !c.SkipUnavailable {
					return err
				}
				info.Unavailable = err.Error()
			} else {
				if _, ok := b.(gpu.Synchronizer); !ok {
					return fmt.Errorf("%s has no completion synchronization", device)
				}
				if device == tensor.Vulkan {
					v, err := vulkan.DeviceInfo()
					if err != nil {
						return err
					}
					info.Name = fmt.Sprintf("%s (Vulkan %d.%d.%d)", v.Name, v.APIVersion>>22, v.APIVersion>>12&1023, v.APIVersion&4095)
				} else {
					v, err := directml.DeviceInfo()
					if err != nil {
						return err
					}
					info.Name = fmt.Sprintf("%s (DirectML 0x%x)", v.Name, v.FeatureLevel)
				}
			}
		}
		info.InitMS = elapsedMS(start)
		r.Devices = append(r.Devices, info)
		if info.Unavailable == "" {
			devices = append(devices, device)
		}
	}
	if len(devices) == 0 {
		return fmt.Errorf("none of the requested devices is available")
	}
	for _, name := range strings.Split(c.Workloads, ",") {
		name = strings.TrimSpace(name)
		ref, err := reference(c, name)
		if err != nil {
			return err
		}
		for _, device := range devices {
			v, err := measure(c, name, device, ref)
			if err != nil {
				return fmt.Errorf("%s/%s: %w", name, device, err)
			}
			r.Results = append(r.Results, v)
		}
	}
	for i := range r.Results {
		for _, cpu := range r.Results {
			if cpu.Device == tensor.CPU && cpu.Workload == r.Results[i].Workload && r.Results[i].MedianMS > 0 {
				r.Results[i].Speedup = cpu.MedianMS / r.Results[i].MedianMS
			}
		}
	}
	if c.JSON {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	fmt.Fprintf(out, "%s %s/%s CPUs=%d GOMAXPROCS=%d FP32 NoGrad batch=%d seq=%d size=%d hidden=%d layers=%d\n", r.GoVersion, r.OS, r.Arch, r.CPUs, r.GOMAXPROCS, c.Batch, c.Sequence, c.Size, c.Hidden, c.Layers)
	for _, d := range r.Devices {
		if d.Unavailable != "" {
			fmt.Fprintf(out, "%s unavailable: %s\n", d.Device, d.Unavailable)
		} else {
			fmt.Fprintf(out, "%s: %s init=%.3fms\n", d.Device, d.Name, d.InitMS)
		}
	}
	fmt.Fprintln(out, "resident synchronized forward + ReleaseGraph; setup, first-forward, validation/readback excluded from median")
	fmt.Fprintln(out, "workload   device      setup_ms   first_ms  median_ms     p95_ms   speedup    max_abs_error       cosine")
	for _, v := range r.Results {
		ratio := "n/a"
		if v.Speedup > 0 {
			ratio = fmt.Sprintf("%.3fx", v.Speedup)
		}
		fmt.Fprintf(out, "%-10s %-9s %10.3f %10.3f %10.3f %10.3f %9s %16.8g %12.9f\n", v.Workload, v.Device, v.SetupMS, v.FirstForwardMS, v.MedianMS, v.P95MS, ratio, v.MaxAbsError, v.Cosine)
	}
	return nil
}

func validate(c config) error {
	for _, v := range []int{c.Size, c.Hidden, c.Heads, c.Layers, c.Batch, c.Sequence, c.Channels} {
		if v < 1 || v > 8192 {
			return fmt.Errorf("shape parameters must be in 1..8192")
		}
	}
	if c.Size%c.Heads != 0 {
		return fmt.Errorf("size must be divisible by heads")
	}
	if c.Warmup < 0 || c.Warmup > 10000 || c.Iterations < 1 || c.Iterations > 10000 {
		return fmt.Errorf("warmup must be 0..10000 and iters 1..10000")
	}
	for _, list := range []struct {
		value string
		valid []string
	}{{c.Devices, []string{"cpu", "vulkan", "directml"}}, {c.Workloads, []string{"gemm", "projection", "ffn", "encoder", "gelu", "layernorm", "bias_gelu", "bias_residual"}}} {
		seen := map[string]bool{}
		for _, s := range strings.Split(list.value, ",") {
			s = strings.TrimSpace(s)
			if !slices.Contains(list.valid, s) || seen[s] {
				return fmt.Errorf("invalid or duplicate selection %q", s)
			}
			seen[s] = true
		}
	}
	for _, shape := range [][]int{{c.Size, c.Size}, {c.Hidden, c.Size}, {c.Batch, c.Sequence, c.Hidden}, {c.Batch, c.Heads, c.Sequence, c.Sequence}, {c.Size, c.Channels, 3}} {
		if _, err := gpu.Elements(shape...); err != nil {
			return err
		}
	}
	return nil
}

func build(c config, name string, device tensor.Device) (_ *workload, err error) {
	w := &workload{}
	defer func() {
		if err != nil {
			w.close()
		}
	}()
	context := autograd.NewExecutionContext()
	rng := rand.New(rand.NewSource(731))
	newTensor := func(shape []int, scale float32) (*autograd.Tensor, error) {
		n, err := gpu.Elements(shape...)
		if err != nil {
			return nil, err
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = (rng.Float32()*2 - 1) * scale
		}
		v, err := context.New(data, shape, device, true)
		if err == nil {
			w.leaves = append(w.leaves, v)
		}
		return v, err
	}
	var forward func() *autograd.Tensor
	switch name {
	case "gemm", "projection":
		rows := c.Size
		if name == "projection" {
			rows = c.Batch * c.Sequence
		}
		a, err := newTensor([]int{rows, c.Size}, 1)
		if err != nil {
			return nil, err
		}
		b, err := newTensor([]int{c.Size, c.Size}, 1/float32(math.Sqrt(float64(c.Size))))
		if err != nil {
			return nil, err
		}
		forward = func() *autograd.Tensor { return autograd.MatMul(a, b) }
	case "ffn":
		x, err := newTensor([]int{c.Batch, c.Sequence, c.Size}, 1)
		if err != nil {
			return nil, err
		}
		w1, err := newTensor([]int{c.Hidden, c.Size}, 1/float32(math.Sqrt(float64(c.Size))))
		if err != nil {
			return nil, err
		}
		b1, err := newTensor([]int{c.Hidden}, .1)
		if err != nil {
			return nil, err
		}
		w2, err := newTensor([]int{c.Size, c.Hidden}, 1/float32(math.Sqrt(float64(c.Hidden))))
		if err != nil {
			return nil, err
		}
		b2, err := newTensor([]int{c.Size}, .1)
		if err != nil {
			return nil, err
		}
		forward = func() *autograd.Tensor {
			h := autograd.BiasGELU(autograd.MatMul(x, autograd.Transpose(w1, 0, 1)), b1)
			return autograd.Add(autograd.MatMul(h, autograd.Transpose(w2, 0, 1)), b2)
		}
	case "gelu":
		x, err := newTensor([]int{c.Batch, c.Sequence, c.Hidden}, 3)
		if err != nil {
			return nil, err
		}
		forward = func() *autograd.Tensor { return autograd.GELU(x, true) }
	case "bias_gelu", "bias_residual":
		channels := c.Size
		if name == "bias_gelu" {
			channels = c.Hidden
		}
		x, err := newTensor([]int{c.Batch, c.Sequence, channels}, 1)
		if err != nil {
			return nil, err
		}
		bias, err := newTensor([]int{channels}, .1)
		if err != nil {
			return nil, err
		}
		if name == "bias_gelu" {
			forward = func() *autograd.Tensor { return autograd.BiasGELU(x, bias) }
		} else {
			residual, err := newTensor([]int{c.Batch, c.Sequence, channels}, 1)
			if err != nil {
				return nil, err
			}
			forward = func() *autograd.Tensor { return autograd.BiasResidual(x, bias, residual) }
		}
	case "layernorm":
		x, err := newTensor([]int{c.Batch, c.Sequence, c.Size}, 1)
		if err != nil {
			return nil, err
		}
		weight, err := newTensor([]int{c.Size}, 1)
		if err != nil {
			return nil, err
		}
		bias, err := newTensor([]int{c.Size}, .1)
		if err != nil {
			return nil, err
		}
		forward = func() *autograd.Tensor { return autograd.LayerNorm(x, weight, bias, 1e-5) }
	case "encoder":
		x, err := newTensor([]int{c.Batch, c.Sequence, c.Channels}, 1)
		if err != nil {
			return nil, err
		}
		cw, err := newTensor([]int{c.Size, c.Channels, 3}, 1/float32(math.Sqrt(float64(c.Channels*3))))
		if err != nil {
			return nil, err
		}
		cb, err := newTensor([]int{c.Size}, .1)
		if err != nil {
			return nil, err
		}
		layers := make([]*autograd.TransformerEncoderLayer, c.Layers)
		for i := range layers {
			layer, err := autograd.NewTransformerEncoderLayer(c.Size, c.Heads, c.Hidden, device)
			if err != nil {
				return nil, err
			}
			for _, p := range layer.NamedParameters() {
				w.leaves = append(w.leaves, p.Value)
			}
			if err := context.BindModule(&layer.Module); err != nil {
				return nil, err
			}
			layer.Training, layer.Attention.Training = false, false
			layers[i] = layer
		}
		forward = func() *autograd.Tensor {
			h := autograd.Conv1dGEMMStride(x, cw, cb, 1, 1, 1)
			for _, layer := range layers {
				h = layer.Forward(h, nil, 0)
			}
			return h
		}
	default:
		return nil, fmt.Errorf("unknown workload %q", name)
	}
	w.forward = func() *autograd.Tensor { var y *autograd.Tensor; context.NoGrad(func() { y = forward() }); return y }
	return w, nil
}

func reference(c config, name string) (values []float32, err error) {
	defer recoverError(&err)
	w, err := build(c, name, tensor.CPU)
	if err != nil {
		return nil, err
	}
	defer w.close()
	y := w.forward()
	defer y.ReleaseGraph()
	return y.ToHost()
}

func measure(c config, name string, device tensor.Device, want []float32) (_ result, err error) {
	defer recoverError(&err)
	releaseCaches(device)
	start := clockNow()
	w, err := build(c, name, device)
	if err != nil {
		return result{}, err
	}
	defer w.close()
	r := result{Workload: name, Device: device, SetupMS: elapsedMS(start)}
	sync := func() error { return nil }
	if device != tensor.CPU {
		b, err := gpu.Get(string(device))
		if err != nil {
			return r, err
		}
		sync = b.(gpu.Synchronizer).Synchronize
	}
	start = clockNow()
	y := w.forward()
	defer y.ReleaseGraph()
	if err := sync(); err != nil {
		return r, err
	}
	r.FirstForwardMS = elapsedMS(start)
	start = clockNow()
	got, err := y.ToHost()
	r.ReadbackMS = elapsedMS(start)
	if err != nil {
		return r, err
	}
	r.MaxAbsError, r.Cosine, err = agreement(got, want)
	if err != nil {
		return r, err
	}
	y.ReleaseGraph()
	for i := 0; i < c.Warmup; i++ {
		y := w.forward()
		if err := sync(); err != nil {
			y.ReleaseGraph()
			return r, err
		}
		y.ReleaseGraph()
	}
	runtime.GC() // Setup/compilation garbage is outside the steady-state samples.
	baseline := snapshot(device)
	for i := 0; i < c.Iterations; i++ {
		start := clockNow()
		y := w.forward()
		if err := sync(); err != nil {
			y.ReleaseGraph()
			return r, err
		}
		y.ReleaseGraph()
		r.SamplesMS = append(r.SamplesMS, elapsedMS(start))
	}
	r.MedianMS, r.P95MS = summary(r.SamplesMS)
	r.Memory = snapshot(device)
	if baseline != nil && (r.Memory.LiveBytes != baseline.LiveBytes || r.Memory.LiveBuffers != baseline.LiveBuffers) {
		return r, fmt.Errorf("live GPU memory increased across measured iterations")
	}
	// Check the final steady-state result too; validation never enters timing.
	y = w.forward()
	defer y.ReleaseGraph()
	if err := sync(); err != nil {
		return r, err
	}
	got, err = y.ToHost()
	if err != nil {
		return r, err
	}
	e, cos, err := agreement(got, want)
	r.MaxAbsError = max(r.MaxAbsError, e)
	r.Cosine = min(r.Cosine, cos)
	return r, err
}

func recoverError(err *error) {
	if r := recover(); r != nil {
		if e, ok := r.(error); ok {
			*err = e
		} else {
			*err = fmt.Errorf("inference panic: %v", r)
		}
	}
}
func summary(samples []float64) (median, p95 float64) {
	v := append([]float64(nil), samples...)
	slices.Sort(v)
	n := len(v)
	median = v[n/2]
	if n%2 == 0 {
		median = (v[n/2-1] + v[n/2]) / 2
	}
	p95 = v[int(math.Ceil(.95*float64(n)))-1]
	return
}
func agreement(got, want []float32) (maxAbs, cosine float64, err error) {
	if len(got) != len(want) || len(got) == 0 {
		return 0, 0, fmt.Errorf("output length mismatch or empty output")
	}
	var dot, xx, yy float64
	for i, v := range got {
		ref := want[i]
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(ref)) || math.IsInf(float64(ref), 0) {
			return 0, 0, fmt.Errorf("non-finite output at %d", i)
		}
		diff := math.Abs(float64(v) - float64(ref))
		maxAbs = max(maxAbs, diff)
		if diff > 1e-4+1e-4*math.Abs(float64(ref)) {
			return maxAbs, 0, fmt.Errorf("CPU agreement failed at %d: %g != %g", i, v, ref)
		}
		dot += float64(v) * float64(ref)
		xx += float64(v) * float64(v)
		yy += float64(ref) * float64(ref)
	}
	if xx == 0 && yy == 0 {
		return maxAbs, 1, nil
	}
	if xx == 0 || yy == 0 {
		return maxAbs, 0, fmt.Errorf("zero-norm mismatch")
	}
	cosine = dot / math.Sqrt(xx*yy)
	if cosine < .999 {
		return maxAbs, cosine, fmt.Errorf("cosine agreement failed: %g", cosine)
	}
	return maxAbs, cosine, nil
}
func releaseCaches(device tensor.Device) {
	if device == tensor.Vulkan {
		vulkan.ReleasePool()
	}
	if device == tensor.DirectML {
		directml.ReleasePool()
		directml.ReleaseOperators()
	}
}
func snapshot(device tensor.Device) *memory {
	if device == tensor.Vulkan {
		s := vulkan.MemoryStats()
		return &memory{LiveBytes: s.LiveBytes, CachedBytes: s.CachedBytes, ReservedBytes: s.ReservedBytes, LiveBuffers: s.LiveBuffers}
	}
	if device == tensor.DirectML {
		s := directml.MemoryStats()
		op := directml.CachedOperatorStats()
		return &memory{LiveBytes: s.LiveBytes, CachedBytes: s.CachedBytes, ReservedBytes: s.ReservedBytes, LiveBuffers: s.LiveBuffers, Operators: op.CachedOperators, OperatorBytes: op.PersistentBytes + op.TemporaryBytes}
	}
	return nil
}
