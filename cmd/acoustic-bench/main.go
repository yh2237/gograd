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
	graph := flag.Bool("graph", false, "capture and time one fixed-shape CUDA training-step replay")
	bf16 := flag.Bool("bf16", false, "BF16 tensor-core GEMM with FP32 weights")
	memoryStats := flag.Bool("memory-stats", false, "print CUDA pool memory after the step")
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
	if *bf16 {
		if device != tensor.CUDA {
			panic("BF16 autocast requires CUDA")
		}
		autograd.BF16Autocast = true
		defer func() { autograd.BF16Autocast = false }()
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
	if *graph {
		if device != tensor.CUDA {
			panic("graph capture requires CUDA")
		}
		if e := m.SetCUDAIndices(ids, spk); e != nil {
			panic(e)
		}
		defer m.ClearCUDAIndices()
	}
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
	if *memoryStats && device == tensor.CUDA {
		s := cuda.MemoryStats()
		fmt.Printf("cuda memory: live=%d cached=%d reserved=%d peak_reserved=%d cache_limit=%d\n", s.LiveBytes, s.CachedBytes, s.ReservedBytes, s.PeakReservedBytes, s.CacheLimitBytes)
	}
	if *graph {
		if e := opt.PrepareGraph(); e != nil {
			panic(e)
		}
		stream, e := cuda.NewStream()
		if e != nil {
			panic(e)
		}
		defer stream.Destroy()
		if e := autograd.SetBLASStream(stream); e != nil {
			panic(e)
		}
		defer autograd.SetBLASStream(nil)
		var capturedLoss *autograd.Tensor
		captured, e := cuda.Capture(stream, func() error {
			opt.ZeroGrad()
			pred := m.Forward(ids, cont, spk)
			loss := autograd.MaskedLoss(pred, target, false)
			loss.RetainGrad()
			capturedLoss = loss
			if e := loss.Backward(); e != nil {
				return e
			}
			autograd.ClipGradNorm(m.Params, 1)
			opt.Step()
			loss.ReleaseGraph()
			return nil
		})
		if e != nil {
			fmt.Printf("gograd cuda graph capture failed: %v\n", e)
			return
		}
		defer captured.Close()
		start := time.Now()
		if e := captured.Launch(); e != nil {
			panic(e)
		}
		submission := time.Since(start)
		if e := stream.Synchronize(); e != nil {
			panic(e)
		}
		synchronized := time.Since(start)
		lossValue, e := capturedLoss.ToHost()
		if e != nil {
			panic(e)
		}
		elapsed := time.Since(start)
		stepCount, e := opt.GraphStepCount()
		if e != nil {
			panic(e)
		}
		fmt.Printf("gograd cuda graph replay: %.3f ms (submit %.3f, wait %.3f, readback %.3f; device AdamW step %d, loss %.6f)\n", float64(elapsed.Microseconds())/1000, float64(submission.Microseconds())/1000, float64((synchronized-submission).Microseconds())/1000, float64((elapsed-synchronized).Microseconds())/1000, stepCount, lossValue[0])
		capturedLoss.Close()
	}
}
func float32NaN() float32 { return float32(math.NaN()) }
