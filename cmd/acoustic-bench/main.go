package main

import (
	"flag"
	"fmt"
	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
	"math"
	"math/rand"
	"time"
)

func main() {
	deviceFlag := flag.String("device", "cpu", "cpu or cuda")
	flag.Parse()
	device := tensor.Device(*deviceFlag)
	m, e := autograd.NewAcoustic(384, 102, 88, []int{1, 2, 4, 8, 16, 1, 2, 4, 8, 16}, device)
	if e != nil {
		panic(e)
	}
	rng := rand.New(rand.NewSource(733))
	for _, p := range m.Params {
		for i := range p.Value.Data {
			p.Value.Data[i] = float32(rng.NormFloat64() * .02)
		}
		if len(p.Name) >= 5 && p.Name[:5] == "norms" && p.Name[len(p.Name)-6:] == "weight" {
			for i := range p.Value.Data {
				p.Value.Data[i] = 1
			}
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
	start := time.Now()
	pred := m.Forward(ids, cont, spk)
	loss := autograd.MaskedLoss(pred, target, false)
	if e := loss.Backward(); e != nil {
		panic(e)
	}
	autograd.ClipGradNorm(m.Params, 1)
	opt.Step()
	fmt.Printf("gograd %s: %.3f ms loss %.6f\n", device, float64(time.Since(start).Microseconds())/1000, loss.Data[0])
}
func float32NaN() float32 { return float32(math.NaN()) }
