package main

import (
	"flag"
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func main() {
	dev := flag.String("device", "cpu", "cpu or cuda")
	warmup := flag.Int("warmup", 1, "untimed training steps")
	gpuProfile := flag.Bool("gpu-profile", false, "report CUDA kernel timings for measured step")
	flag.Parse()
	device := tensor.Device(*dev)
	if device == tensor.CUDA {
		ctx, e := autograd.NewCUDAContext()
		if e != nil {
			panic(e)
		}
		defer ctx.Close()
	}
	layers := make([]*autograd.TransformerEncoderLayer, 4)
	model := &autograd.Module{}
	for i := range layers {
		layer, e := autograd.NewTransformerEncoderLayer(256, 4, 1024, device)
		if e != nil {
			panic(e)
		}
		layers[i] = layer
		model.Children = append(model.Children, autograd.NamedModule{Name: fmt.Sprintf("%d", i), Module: &layer.Module})
	}
	rng := rand.New(rand.NewSource(731))
	data := make([]float32, 16*256*256)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	x, e := autograd.New(data, []int{16, 256, 256}, device, false)
	if e != nil {
		panic(e)
	}
	defer x.Close()
	params := model.NamedParameters()
	opt := autograd.NewAdamW(params, .001, .01)
	syncDevice := func() {
		if device == tensor.CUDA {
			if e := cuda.Synchronize(); e != nil {
				panic(e)
			}
		}
	}
	for step := 0; step <= *warmup; step++ {
		if step == *warmup && *gpuProfile {
			autograd.EnableGPUProfile(true)
		}
		start := time.Now()
		opt.ZeroGrad()
		y := x
		for i, layer := range layers {
			y = layer.Forward(y, nil, uint32(step*10+i*3))
		}
		loss := autograd.Mean(autograd.Mul(y, y), 0, 1, 2)
		if e := loss.Backward(); e != nil {
			panic(e)
		}
		autograd.ClipGradNorm(params, 1)
		opt.Step()
		loss.ReleaseGraph()
		syncDevice()
		if step == *warmup {
			fmt.Printf("gograd %s transformer: %.3f ms\n", device, float64(time.Since(start).Microseconds())/1000)
			if *gpuProfile {
				times := autograd.GPUProfile()
				names := make([]string, 0, len(times))
				for name := range times {
					names = append(names, name)
				}
				sort.Slice(names, func(i, j int) bool { return times[names[i]].Duration > times[names[j]].Duration })
				for _, name := range names {
					v := times[name]
					fmt.Printf("  %-22s %7.3f ms %d calls\n", name, float64(v.Duration.Microseconds())/1000, v.Count)
				}
			}
		}
	}
}
