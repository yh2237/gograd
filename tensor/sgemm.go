package tensor

import (
	"runtime"
	"sync"
)

// SGEMM computes C = A*B for contiguous row-major matrices using 4x8
// register tiles, parallelized over row ranges.
func SGEMM(c, a, b []float32, m, n, k int) {
	if len(a) != m*k || len(b) != k*n || len(c) != m*n {
		panic("tensor: SGEMM shape mismatch")
	}
	const rowBlock = 16
	tiles := (m + rowBlock - 1) / rowBlock
	workers := min(runtime.GOMAXPROCS(0), tiles)
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
				end := min(row+rowBlock, m)
				for i := row; i < end; i += 4 {
					r := min(4, end-i)
					for j := 0; j < n; j += 8 {
						q := min(8, n-j)
						if r != 4 || q != 8 {
							for ii := 0; ii < r; ii++ {
								for jj := 0; jj < q; jj++ {
									var sum float32
									for d := 0; d < k; d++ {
										sum += a[(i+ii)*k+d] * b[d*n+j+jj]
									}
									c[(i+ii)*n+j+jj] = sum
								}
							}
							continue
						}
						ar0 := a[(i+0)*k : (i+0+1)*k]
						ar1 := a[(i+1)*k : (i+1+1)*k]
						ar2 := a[(i+2)*k : (i+2+1)*k]
						ar3 := a[(i+3)*k : (i+3+1)*k]
						var c00, c01, c02, c03, c04, c05, c06, c07, c10, c11, c12, c13, c14, c15, c16, c17, c20, c21, c22, c23, c24, c25, c26, c27, c30, c31, c32, c33, c34, c35, c36, c37 float32
						for d := 0; d < k; d++ {
								br := b[d*n+j:]
							b0 := br[0]
							b1 := br[1]
							b2 := br[2]
							b3 := br[3]
							b4 := br[4]
							b5 := br[5]
							b6 := br[6]
							b7 := br[7]
							a0 := ar0[d]
							a1 := ar1[d]
							a2 := ar2[d]
							a3 := ar3[d]
							c00 += a0 * b0
							c01 += a0 * b1
							c02 += a0 * b2
							c03 += a0 * b3
							c04 += a0 * b4
							c05 += a0 * b5
							c06 += a0 * b6
							c07 += a0 * b7
							c10 += a1 * b0
							c11 += a1 * b1
							c12 += a1 * b2
							c13 += a1 * b3
							c14 += a1 * b4
							c15 += a1 * b5
							c16 += a1 * b6
							c17 += a1 * b7
							c20 += a2 * b0
							c21 += a2 * b1
							c22 += a2 * b2
							c23 += a2 * b3
							c24 += a2 * b4
							c25 += a2 * b5
							c26 += a2 * b6
							c27 += a2 * b7
							c30 += a3 * b0
							c31 += a3 * b1
							c32 += a3 * b2
							c33 += a3 * b3
							c34 += a3 * b4
							c35 += a3 * b5
							c36 += a3 * b6
							c37 += a3 * b7
						}
						p0 := (i+0)*n + j
						c[p0+0] = c00
						c[p0+1] = c01
						c[p0+2] = c02
						c[p0+3] = c03
						c[p0+4] = c04
						c[p0+5] = c05
						c[p0+6] = c06
						c[p0+7] = c07
						p1 := (i+1)*n + j
						c[p1+0] = c10
						c[p1+1] = c11
						c[p1+2] = c12
						c[p1+3] = c13
						c[p1+4] = c14
						c[p1+5] = c15
						c[p1+6] = c16
						c[p1+7] = c17
						p2 := (i+2)*n + j
						c[p2+0] = c20
						c[p2+1] = c21
						c[p2+2] = c22
						c[p2+3] = c23
						c[p2+4] = c24
						c[p2+5] = c25
						c[p2+6] = c26
						c[p2+7] = c27
						p3 := (i+3)*n + j
						c[p3+0] = c30
						c[p3+1] = c31
						c[p3+2] = c32
						c[p3+3] = c33
						c[p3+4] = c34
						c[p3+5] = c35
						c[p3+6] = c36
						c[p3+7] = c37
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
