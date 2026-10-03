package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// Compare materialized GEMM attention with tiled online-softmax attention.
func main() {
	seq := flag.Int("seq", 256, "query and key sequence length")
	heads := flag.Int("heads", 4, "heads")
	dim := flag.Int("dim", 64, "head dimension")
	algorithm := flag.String("algorithm", "flash", "flash or materialized")
	warmup := flag.Int("warmup", 1, "untimed steps")
	flag.Parse()
	if *algorithm != "flash" && *algorithm != "materialized" {
		panic("invalid algorithm")
	}
	ctx, e := autograd.NewCUDAContext()
	if e != nil {
		panic(e)
	}
	defer ctx.Close()
	autograd.AttentionAlgorithm = *algorithm
	defer func() { autograd.AttentionAlgorithm = "auto" }()
	size := *heads * *seq * *dim
	rng := rand.New(rand.NewSource(147))
	data := func() []float32 {
		a := make([]float32, size)
		for i := range a {
			a[i] = float32(rng.NormFloat64()) * .1
		}
		return a
	}
	shape := []int{1, *heads, *seq, *dim}
	q, _ := autograd.New(data(), shape, tensor.CUDA, true)
	k, _ := autograd.New(data(), shape, tensor.CUDA, true)
	v, _ := autograd.New(data(), shape, tensor.CUDA, true)
	up, _ := autograd.New(data(), shape, tensor.CUDA, false)
	defer q.Close()
	defer k.Close()
	defer v.Close()
	defer up.Close()
	step := func(measure bool) {
		q.ZeroGrad()
		k.ZeroGrad()
		v.ZeroGrad()
		if measure {
			cuda.ResetAllocationPeak()
		}
		baseline := cuda.MemoryStats().LiveBytes
		startEvent, e := cuda.NewEvent(true)
		if e != nil {
			panic(e)
		}
		defer startEvent.Destroy()
		forwardEvent, e := cuda.NewEvent(true)
		if e != nil {
			panic(e)
		}
		defer forwardEvent.Destroy()
		endEvent, e := cuda.NewEvent(true)
		if e != nil {
			panic(e)
		}
		defer endEvent.Destroy()
		if e := startEvent.Record(nil); e != nil {
			panic(e)
		}
		y := autograd.ScaledDotProductAttention(q, k, v, nil)
		if e := forwardEvent.Record(nil); e != nil {
			panic(e)
		}
		loss := autograd.Sum(autograd.Mul(y, up), 0, 1, 2, 3)
		if e := loss.Backward(); e != nil {
			panic(e)
		}
		if e := endEvent.Record(nil); e != nil {
			panic(e)
		}
		if e := endEvent.Synchronize(); e != nil {
			panic(e)
		}
		if measure {
			forward, e := cuda.ElapsedTime(startEvent, forwardEvent)
			if e != nil {
				panic(e)
			}
			elapsed, e := cuda.ElapsedTime(startEvent, endEvent)
			if e != nil {
				panic(e)
			}
			stats := cuda.MemoryStats()
			fmt.Printf("%s seq=%d heads=%d dim=%d gpu_forward=%.3f ms gpu_step=%.3f ms peak_extra=%.2f MiB\n", *algorithm, *seq, *heads, *dim, forward, elapsed, float64(stats.PeakLiveBytes-baseline)/1048576)
		}
		loss.ReleaseGraph()
	}
	for i := 0; i < *warmup; i++ {
		step(false)
	}
	step(true)
}
