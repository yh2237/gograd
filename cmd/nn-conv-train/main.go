// Command nn-conv-train trains a small residual Conv1d model on synthetic
// variable-length sequences and prints each full training-step duration.
package main

import (
	"flag"
	"fmt"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/nn"
	"github.com/yh2237/gograd/tensor"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

func main() {
	deviceFlag := flag.String("device", "cpu", "cpu or cuda")
	hidden := flag.Int("hidden", 128, "hidden channels")
	kernel := flag.Int("kernel", 5, "odd convolution kernel")
	batch := flag.Int("batch", 32, "batch size")
	frames := flag.Int("time", 100, "padded frames")
	steps := flag.Int("steps", 3, "training steps")
	profile := flag.String("cpuprofile", "", "write CPU profile")
	lossKind := flag.String("loss", "l1", "l1 or mse")
	flag.Parse()
	if *profile != "" {
		f, err := os.Create(*profile)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			panic(err)
		}
		defer pprof.StopCPUProfile()
	}
	device := tensor.Device(*deviceFlag)
	if device != tensor.CPU && device != tensor.CUDA {
		panic("invalid device")
	}
	if device == tensor.CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if !cuda.Available() {
			panic("CUDA unavailable")
		}
		if e := cuda.SetDevice(0); e != nil {
			panic(e)
		}
	}
	var blas *cuda.Blas
	if device == tensor.CUDA {
		var e error
		blas, e = cuda.NewBlas()
		if e != nil {
			panic(e)
		}
		defer blas.Destroy()
	}
	rng := rand.New(rand.NewSource(23))
	input, e := nn.NewLinearOn(device, 74, *hidden, rng)
	if e != nil {
		panic(e)
	}
	modules := []nn.Module{input}
	for _, d := range []int{1, 2, 4, 1, 2, 4} {
		conv, e := nn.NewConv1dOn(device, *hidden, *hidden, *kernel, d, rng)
		if e != nil {
			panic(e)
		}
		modules = append(modules, &nn.Residual{Inner: &nn.Sequential{Modules: []nn.Module{conv, &nn.GELU{}}}})
	}
	last, e := nn.NewLinearOn(device, *hidden, 72, rng)
	if e != nil {
		panic(e)
	}
	modules = append(modules, last)
	model := &nn.Sequential{Modules: modules}
	defer model.Close()
	optimizer, e := nn.NewAdamW(model.Parameters(), model.Gradients(), .001, .01, 1)
	if e != nil {
		panic(e)
	}
	defer optimizer.Close()
	xdata := make([]float32, *batch**frames*74)
	targetData := make([]float32, *batch**frames*72)
	mask := make([]float32, *batch**frames)
	for b := 0; b < *batch; b++ {
		valid := *frames - b%7
		for t := 0; t < valid; t++ {
			mask[b**frames+t] = 1
			for c := 0; c < 74; c++ {
				xdata[(b**frames+t)*74+c] = float32(rng.NormFloat64() * .1)
			}
			for c := 0; c < 72; c++ {
				targetData[(b**frames+t)*72+c] = float32(math.Sin(float64(t)*.03 + float64(c)*.1))
			}
		}
	}
	x, e := tensor.FromHostOn(device, []int{*batch, *frames, 74}, xdata)
	if e != nil {
		panic(e)
	}
	defer x.Close()
	target, e := tensor.FromHostOn(device, []int{*batch, *frames, 72}, targetData)
	if e != nil {
		panic(e)
	}
	defer target.Close()
	if *lossKind != "l1" && *lossKind != "mse" {
		panic("invalid loss")
	}
	lossOp, e := nn.NewMaskedLoss(device, mask, nil, 72, *lossKind == "l1")
	if e != nil {
		panic(e)
	}
	defer lossOp.Close()
	for step := 0; step < *steps; step++ {
		start := time.Now()
		model.ZeroGrad()
		y, e := model.Forward(blas, x)
		if e != nil {
			panic(e)
		}
		var loss float64
		var g *tensor.Tensor
		loss, g, e = lossOp.Forward(y, target)
		if e != nil {
			panic(e)
		}
		dx, e := model.Backward(blas, g)
		if e != nil {
			panic(e)
		}
		dx.Close()
		g.Close()
		model.CloseActivations()
		if e = optimizer.Step(); e != nil {
			panic(e)
		}
		if device == tensor.CUDA {
			if e = cuda.Synchronize(); e != nil {
				panic(e)
			}
		}
		fmt.Printf("device=%s hidden=%d kernel=%d step=%d loss=%.6f duration=%s\n", device, *hidden, *kernel, step, loss, time.Since(start))
	}
}
