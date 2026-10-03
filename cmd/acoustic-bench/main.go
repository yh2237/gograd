package main

import (
	"flag"
	"fmt"
	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"sort"
	"time"
)

func main() {
	deviceFlag := flag.String("device", "cpu", "cpu or cuda")
	warmup := flag.Int("warmup", 0, "untimed steps before measurement")
	profile := flag.String("cpuprofile", "", "CPU profile path")
	gpuProfile := flag.Bool("gpu-profile", false, "synchronize and report CUDA operator timings")
	flag.Parse()
	if *profile != "" {
		f, e := os.Create(*profile)
		if e != nil {
			panic(e)
		}
		defer f.Close()
		if e = pprof.StartCPUProfile(f); e != nil {
			panic(e)
		}
		defer pprof.StopCPUProfile()
	}
	device := tensor.Device(*deviceFlag)
	if device == tensor.CUDA {
		context, e := autograd.NewCUDAContext()
		if e != nil {
			panic(e)
		}
		defer context.Close()
	}
	m, e := autograd.NewAcoustic(384, 102, 88, []int{1, 2, 4, 8, 16, 1, 2, 4, 8, 16}, device)
	if e != nil {
		panic(e)
	}
	rng := rand.New(rand.NewSource(733))
	for _, p := range m.Params {
		values := make([]float32, p.Value.Numel())
		for i := range values {
			values[i] = float32(rng.NormFloat64() * .02)
		}
		if len(p.Name) >= 5 && p.Name[:5] == "norms" && p.Name[len(p.Name)-6:] == "weight" {
			for i := range values {
				values[i] = 1
			}
		}
		if e := p.Value.CopyFrom(values); e != nil {
			panic(e)
		}
	}
	ids := make([]int, 24*400*3)
	for i := range ids {
		ids[i] = rng.Intn(46)
	}
	spk := make([]int, 24)
	for i := range spk {
		spk[i] = rng.Intn(102)
	}
	cv := make([]float32, 24*400*4)
	for i := range cv {
		cv[i] = float32(rng.NormFloat64())
	}
	cont, e := autograd.New(cv, []int{24, 400, 4}, device, false)
	if e != nil {
		panic(e)
	}
	tv := make([]float32, 24*400*88)
	for i := range tv {
		tv[i] = float32(rng.NormFloat64())
	}
	for row := 0; row < 24*400; row++ {
		if row%400 >= 390 {
			for j := 0; j < 88; j++ {
				tv[row*88+j] = float32NaN()
			}
		}
	}
	target, e := autograd.New(tv, []int{24, 400, 88}, device, false)
	if e != nil {
		panic(e)
	}
	opt := autograd.NewAdamW(m.Params, .001, .0001)
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
		opt.ZeroGrad()
		start := time.Now()
		pred := m.Forward(ids, cont, spk)
		loss := autograd.MaskedLoss(pred, target, false)
		syncDevice()
		forward := time.Since(start)
		lossValue, e := loss.ToHost()
		if e != nil {
			panic(e)
		}
		start = time.Now()
		if e := loss.Backward(); e != nil {
			panic(e)
		}
		syncDevice()
		backward := time.Since(start)
		start = time.Now()
		autograd.ClipGradNorm(m.Params, 1)
		syncDevice()
		clip := time.Since(start)
		start = time.Now()
		opt.Step()
		loss.ReleaseGraph()
		syncDevice()
		update := time.Since(start)
		if step == *warmup {
			fmt.Printf("gograd %s: %.3f ms (forward %.3f backward %.3f clip %.3f adamw %.3f) loss %.6f\n", device, float64((forward+backward+clip+update).Microseconds())/1000, float64(forward.Microseconds())/1000, float64(backward.Microseconds())/1000, float64(clip.Microseconds())/1000, float64(update.Microseconds())/1000, lossValue[0])
			if *gpuProfile {
				times := autograd.GPUProfile()
				keys := make([]string, 0, len(times))
				for k := range times {
					keys = append(keys, k)
				}
				sort.Slice(keys, func(i, j int) bool { return times[keys[i]].Duration > times[keys[j]].Duration })
				for _, k := range keys {
					v := times[k]
					fmt.Printf("  %-22s %7.3f ms %d calls\n", k, float64(v.Duration.Microseconds())/1000, v.Count)
				}
			}
		}
	}
}
func float32NaN() float32 { return float32(math.NaN()) }
