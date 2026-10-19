package main

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func smallConfig() config {
	return config{Devices: "cpu", Workloads: "gemm,projection,ffn,encoder", Size: 8, Hidden: 16, Heads: 2, Layers: 2, Batch: 1, Sequence: 5, Channels: 3, Warmup: 0, Iterations: 2}
}
func TestCPUWorkloads(t *testing.T) {
	c := smallConfig()
	if err := validate(c); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"gemm", "projection", "ffn", "encoder", "gelu", "layernorm"} {
		t.Run(name, func(t *testing.T) {
			want, err := reference(c, name)
			if err != nil {
				t.Fatal(err)
			}
			r, err := measure(c, name, tensor.CPU, want)
			if err != nil {
				t.Fatal(err)
			}
			if r.MedianMS <= 0 || r.P95MS < r.MedianMS || len(r.SamplesMS) != 2 || r.Cosine < .999 || r.MaxAbsError != 0 {
				t.Fatal(r)
			}
		})
	}
}
func TestJSONCommand(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"-devices", "cpu", "-workloads", "gemm,ffn", "-size", "8", "-hidden", "16", "-heads", "2", "-layers", "1", "-seq", "5", "-channels", "3", "-warmup", "0", "-iters", "1", "-json"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	var r report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if len(r.Results) != 2 || r.Results[0].Speedup != 1 || r.Results[0].Device != tensor.CPU {
		t.Fatal(r)
	}
}
func TestValidation(t *testing.T) {
	for _, change := range []func(*config){func(c *config) { c.Size = 0 }, func(c *config) { c.Iterations = 0 }, func(c *config) { c.Warmup = -1 }, func(c *config) { c.Devices = "cpu,cpu" }, func(c *config) { c.Devices = "cuda" }, func(c *config) { c.Workloads = "unknown" }, func(c *config) { c.Heads = 3 }} {
		c := smallConfig()
		change(&c)
		if err := validate(c); err == nil {
			t.Fatal("invalid config accepted", c)
		}
	}
}
func TestAgreement(t *testing.T) {
	if _, cos, err := agreement([]float32{1, 2, 3}, []float32{1, 2, 3}); err != nil || math.Abs(cos-1) > 1e-12 {
		t.Fatal(cos, err)
	}
	for _, data := range [][]float32{{1}, {1, 2, 4}, {1, 2, float32(math.NaN())}, {1, 2, float32(math.Inf(1))}} {
		if _, _, err := agreement(data, []float32{1, 2, 3}); err == nil {
			t.Fatal("invalid output accepted")
		}
	}
	if _, cos, err := agreement([]float32{0}, []float32{0}); err != nil || cos != 1 {
		t.Fatal(cos, err)
	}
}
func TestSummary(t *testing.T) {
	median, p95 := summary([]float64{4, 1, 2, 3})
	if median != 2.5 || p95 != 4 {
		t.Fatal(median, p95)
	}
}
