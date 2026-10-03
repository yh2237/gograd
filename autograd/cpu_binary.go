package autograd

import (
	"runtime"
	"sync"
)

// cpuBinaryOp keeps the scalar rule in a switch and accumulates directly into
// gradients when indices are one-to-one. Broadcast gradients use one partial
// per worker, avoiding atomics and full-size temporary gradients.
func cpuBinaryOp(a, b *Tensor, op int) *Tensor {
	same(a, b)
	s := bshape(a.Shape, b.Shape)
	if len(s) > 3 {
		spec := elementwiseRegistry[[]string{"add", "sub", "mul", "div"}[op]]
		return binary(a, b, spec.CPUForward, spec.CPUGradA, spec.CPUGradB)
	}
	ai, bi := makeBroadcastIndex(s, a.Shape), makeBroadcastIndex(s, b.Shape)
	out := cpuAlloc(numel(s))
	parallelFor(len(out), func(start, end int) {
		for i := start; i < end; i++ {
			x, y := a.Data[ai.at(i)], b.Data[bi.at(i)]
			switch op {
			case 0:
				out[i] = x + y
			case 1:
				out[i] = x - y
			case 2:
				out[i] = x * y
			case 3:
				out[i] = x / y
			}
		}
	})
	return result(out, s, []*Tensor{a, b}, func(g []float32) {
		if !a.RequiresGrad && !b.RequiresGrad {
			return
		}
		if a.RequiresGrad && a.Grad == nil {
			a.Grad = make([]float32, a.Numel())
		}
		if b.RequiresGrad && b.Grad == nil {
			b.Grad = make([]float32, b.Numel())
		}
		workers := min(runtime.GOMAXPROCS(0), len(g)/65536)
		if workers < 1 {
			workers = 1
		}
		localsA, localsB := make([][]float32, workers), make([][]float32, workers)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			start, end := len(g)*w/workers, len(g)*(w+1)/workers
			wg.Add(1)
			go func(worker, start, end int) {
				defer wg.Done()
				var da, db []float32
				if a.RequiresGrad {
					da = a.Grad
					if ai.kind != 1 && workers > 1 {
						da = cpuAlloc(a.Numel())
						localsA[worker] = da
					}
				}
				if b.RequiresGrad {
					db = b.Grad
					if bi.kind != 1 && workers > 1 {
						db = cpuAlloc(b.Numel())
						localsB[worker] = db
					}
				}
				for i := start; i < end; i++ {
					ia, ib := ai.at(i), bi.at(i)
					up := g[i]
					switch op {
					case 0:
						if da != nil {
							da[ia] += up
						}
						if db != nil {
							db[ib] += up
						}
					case 1:
						if da != nil {
							da[ia] += up
						}
						if db != nil {
							db[ib] -= up
						}
					case 2:
						if da != nil {
							da[ia] += up * b.Data[ib]
						}
						if db != nil {
							db[ib] += up * a.Data[ia]
						}
					case 3:
						if da != nil {
							da[ia] += up / b.Data[ib]
						}
						if db != nil {
							y := b.Data[ib]
							db[ib] -= up * a.Data[ia] / (y * y)
						}
					}
				}
			}(w, start, end)
		}
		wg.Wait()
		for _, local := range localsA {
			if local != nil {
				for i, v := range local {
					a.Grad[i] += v
				}
				cpuRelease(local)
			}
		}
		for _, local := range localsB {
			if local != nil {
				for i, v := range local {
					b.Grad[i] += v
				}
				cpuRelease(local)
			}
		}
	})
}
