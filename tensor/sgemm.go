package tensor

import (
	"runtime"
	"sync"
)

// SGEMM computes C = A*B for contiguous row-major matrices. It splits row
// tiles across all available Go workers and blocks the shared and column axes.
func SGEMM(c, a, b []float32, m, n, k int) {
	if len(a) != m*k || len(b) != k*n || len(c) != m*n {
		panic("tensor: SGEMM shape mismatch")
	}
	clear(c)
	const rowBlock, colBlock, depthBlock = 32, 64, 64
	workers := runtime.GOMAXPROCS(0)
	tiles := (m + rowBlock - 1) / rowBlock
	if workers > tiles {
		workers = tiles
	}
	if workers == 0 {
		return
	}
	jobs := make(chan int, tiles)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for row := range jobs {
				rEnd := min(row+rowBlock, m)
				for col := 0; col < n; col += colBlock {
					cEnd := min(col+colBlock, n)
					for depth := 0; depth < k; depth += depthBlock {
						dEnd := min(depth+depthBlock, k)
						for i := row; i < rEnd; i++ {
							cr := c[i*n:]
							ar := a[i*k:]
							for d := depth; d < dEnd; d++ {
								v := ar[d]
								br := b[d*n:]
								for j := col; j < cEnd; j++ {
									cr[j] += v * br[j]
								}
							}
						}
					}
				}
			}
		}()
	}
	for row := 0; row < m; row += rowBlock {
		jobs <- row
	}
	close(jobs)
	wg.Wait()
}
