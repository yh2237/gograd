package autograd

import (
	"runtime"
	"sync"
)

// binaryScalar is small enough to inline, which keeps the elementwise loops
// free of indirect calls.
func binaryScalar(op int, x, y float32) float32 {
	switch op {
	case 0:
		return x + y
	case 1:
		return x - y
	case 2:
		return x * y
	case 3:
		return x / y
	case 4:
		return max(x, y)
	case 5:
		return min(x, y)
	}
	panic("autograd: unknown binary op")
}

// binaryGrad splits the incoming gradient between the operands. For max and min
// the comparison is strict on the left, so ties send the whole gradient right
// and no element is counted twice.
func binaryGrad(op int, up, x, y float32) (float32, float32) {
	switch op {
	case 0:
		return up, up
	case 1:
		return up, -up
	case 2:
		return up * y, up * x
	case 3:
		return up / y, -up * x / (y * y)
	case 4:
		if x > y {
			return up, 0
		}
		return 0, up
	case 5:
		if x < y {
			return up, 0
		}
		return 0, up
	}
	panic("autograd: unknown binary op")
}

// cpuBinaryOp accumulates directly into gradients when the indices are
// one-to-one. Broadcast gradients use one partial per worker, avoiding atomics
// and full-size temporary gradients.
func cpuBinaryOp(a, b *Tensor, op int) *Tensor {
	same(a, b)
	s := bshape(a.Shape, b.Shape)
	ai, bi := makeBroadcastIndex(s, a.Shape), makeBroadcastIndex(s, b.Shape)
	identity := ai.kind == 1 && bi.kind == 1
	// A right operand that repeats along the last axis is the common bias case;
	// a counter replaces the per-element modulo.
	lastAxis := ai.kind == 1 && bi.kind == 3
	out := cpuAllocUninit(numel(s))
	switch {
	case identity:
		parallelFor(len(out), func(start, end int) {
			for i := start; i < end; i++ {
				out[i] = binaryScalar(op, a.Data[i], b.Data[i])
			}
		})
	case lastAxis:
		m := bi.d2
		parallelFor(len(out), func(start, end int) {
			j := start % m
			for i := start; i < end; i++ {
				out[i] = binaryScalar(op, a.Data[i], b.Data[j])
				if j++; j == m {
					j = 0
				}
			}
		})
	default:
		parallelFor(len(out), func(start, end int) {
			for i := start; i < end; i++ {
				out[i] = binaryScalar(op, a.Data[ai.at(i)], b.Data[bi.at(i)])
			}
		})
	}
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
		if workers < 2 {
			workers = 1
		}
		localsA, localsB := make([][]float32, workers), make([][]float32, workers)
		run := func(worker, start, end int) {
			var da, db []float32
			// Only a broadcast operand needs a per-worker partial; an identity
			// operand writes disjoint indices, so the shared gradient is safe.
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
			if lastAxis {
				m := bi.d2
				j := start % m
				for i := start; i < end; i++ {
					ga, gb := binaryGrad(op, g[i], a.Data[i], b.Data[j])
					if da != nil {
						da[i] += ga
					}
					if db != nil {
						db[j] += gb
					}
					if j++; j == m {
						j = 0
					}
				}
				return
			}
			for i := start; i < end; i++ {
				ia, ib := i, i
				if !identity {
					ia, ib = ai.at(i), bi.at(i)
				}
				ga, gb := binaryGrad(op, g[i], a.Data[ia], b.Data[ib])
				if da != nil {
					da[ia] += ga
				}
				if db != nil {
					db[ib] += gb
				}
			}
		}
		if workers == 1 {
			run(0, 0, len(g))
		} else {
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					run(w, len(g)*w/workers, len(g)*(w+1)/workers)
				}()
			}
			wg.Wait()
		}
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
