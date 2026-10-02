package tensor

import (
	"runtime"
	"sync"
)

// SGEMM computes C = A*B for contiguous row-major matrices.
func SGEMM(c, a, b []float32, m, n, k int) {
	SGEMMOp(c, a, b, m, n, k, false, false)
}

// SGEMMOp computes C = op(A)*op(B), with row-major stored operands.
// transA stores A as k-by-m; transB stores B as n-by-k.
func SGEMMOp(c, a, b []float32, m, n, k int, transA, transB bool) {
	if len(a) != m*k || len(b) != k*n || len(c) != m*n {
		panic("tensor: SGEMM shape mismatch")
	}
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		clear(c)
		return
	}
	mc := 32
	if m <= 256 && k >= 1024 {
		mc = 16
	}
	const kc = 256
	const nr = 4
	groups := (n + nr - 1) / nr
	bp := make([]float32, groups*k*nr)
	for g := 0; g < groups; g++ {
		j := g * nr
		width := min(n-j, nr)
		panel := bp[g*k*nr : (g+1)*k*nr]
		for d := 0; d < k; d++ {
			if transB {
				for q := 0; q < width; q++ {
					panel[d*nr+q] = b[(j+q)*k+d]
				}
			} else {
				copy(panel[d*nr:d*nr+width], b[d*n+j:d*n+j+width])
			}
		}
	}
	tiles := (m + mc - 1) / mc
	workers := min(runtime.GOMAXPROCS(0), tiles)
	jobs := make(chan int, tiles)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ap := make([]float32, mc*kc)
			for row := range jobs {
				end := min(row+mc, m)
				for depth := 0; depth < k; depth += kc {
					nk := min(kc, k-depth)
					for i := row; i < end; i += 4 {
						height := min(4, end-i)
						panel := ap[(i-row)*kc : (i-row+4)*kc]
						for d := 0; d < nk; d++ {
							for r := 0; r < height; r++ {
								if transA {
									panel[d*4+r] = a[(depth+d)*m+i+r]
								} else {
									panel[d*4+r] = a[(i+r)*k+depth+d]
								}
							}
						}
					}
					for g := 0; g < groups; g++ {
						j := g * nr
						width := min(n-j, nr)
						bb := bp[g*k*nr+depth*nr:]
						for i := row; i < end; i += 4 {
							height := min(4, end-i)
							aa := ap[(i-row)*kc:]
							if height == 4 && width == 4 {
								kernel4x4(c, aa, bb, i*n+j, n, nk, depth != 0)
							} else {
								for r := 0; r < height; r++ {
									for q := 0; q < width; q++ {
										var sum float32
										if depth != 0 {
											sum = c[(i+r)*n+j+q]
										}
										for d := 0; d < nk; d++ {
											sum += aa[d*4+r] * bb[d*4+q]
										}
										c[(i+r)*n+j+q] = sum
									}
								}
							}
						}
					}
				}
			}
		}()
	}
	for row := 0; row < m; row += mc {
		jobs <- row
	}
	close(jobs)
	wg.Wait()
}

func kernel4x4(c, a, b []float32, offset, stride, k int, add bool) {
	var c00 float32
	var c01 float32
	var c02 float32
	var c03 float32
	var c10 float32
	var c11 float32
	var c12 float32
	var c13 float32
	var c20 float32
	var c21 float32
	var c22 float32
	var c23 float32
	var c30 float32
	var c31 float32
	var c32 float32
	var c33 float32
	if add {
		c00 = c[offset+0*stride+0]
		c01 = c[offset+0*stride+1]
		c02 = c[offset+0*stride+2]
		c03 = c[offset+0*stride+3]
		c10 = c[offset+1*stride+0]
		c11 = c[offset+1*stride+1]
		c12 = c[offset+1*stride+2]
		c13 = c[offset+1*stride+3]
		c20 = c[offset+2*stride+0]
		c21 = c[offset+2*stride+1]
		c22 = c[offset+2*stride+2]
		c23 = c[offset+2*stride+3]
		c30 = c[offset+3*stride+0]
		c31 = c[offset+3*stride+1]
		c32 = c[offset+3*stride+2]
		c33 = c[offset+3*stride+3]
	}
	for d := 0; d < k; d++ {
		ar := a[d*4 : d*4+4]
		br := b[d*4 : d*4+4]
		a0 := ar[0]
		a1 := ar[1]
		a2 := ar[2]
		a3 := ar[3]
		b0 := br[0]
		b1 := br[1]
		b2 := br[2]
		b3 := br[3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}
	c[offset+0*stride+0] = c00
	c[offset+0*stride+1] = c01
	c[offset+0*stride+2] = c02
	c[offset+0*stride+3] = c03
	c[offset+1*stride+0] = c10
	c[offset+1*stride+1] = c11
	c[offset+1*stride+2] = c12
	c[offset+1*stride+3] = c13
	c[offset+2*stride+0] = c20
	c[offset+2*stride+1] = c21
	c[offset+2*stride+2] = c22
	c[offset+2*stride+3] = c23
	c[offset+3*stride+0] = c30
	c[offset+3*stride+1] = c31
	c[offset+3*stride+2] = c32
	c[offset+3*stride+3] = c33
}
