package tensor

import (
	"math/bits"
	"os"
	"runtime"
	"sync"
)

const maxSGEMMPoolSize = 1 << 26

var sgemmPools [81]sync.Pool

func sgemmPoolBucket(n int) (int, int) {
	if n <= 64 {
		return 0, 64
	}
	shift := bits.Len(uint(n - 1))
	base := 1 << (shift - 1)
	step := 1 << (shift - 3)
	sub := (n - 1 - base) / step
	return 1 + (shift-7)*4 + sub, base + (sub+1)*step
}

// The 4x16 microkernel needs six loads per eight FMAs, one fewer per FMA than
// 8x8. Set the environment variable to 0 to keep the 4x8 fallback.
var use4x16 = useAVX2FMA && os.Getenv("GOGRAD_SGEMM_4X16") != "0"

// sgemmTile reports the microkernel block size.
func sgemmTile() (mr, nr int) {
	if !useAVX2FMA {
		return 4, 4
	}
	if use4x16 {
		return 6, 16
	}
	return 4, 8
}

type PackedB struct {
	data []float32
	n, k int
}

func PackB(b []float32, n, k int, transB bool) *PackedB {
	if len(b) != n*k {
		panic("tensor: SGEMM shape mismatch")
	}
	_, nr := sgemmTile()
	groups := (n + nr - 1) / nr
	packed := &PackedB{data: make([]float32, groups*k*nr), n: n, k: k}
	for group := 0; group < groups; group++ {
		column := group * nr
		width := min(n-column, nr)
		panel := packed.data[group*k*nr : (group+1)*k*nr]
		for depth := 0; depth < k; depth++ {
			if transB {
				for offset := 0; offset < width; offset++ {
					panel[depth*nr+offset] = b[(column+offset)*k+depth]
				}
			} else {
				copy(panel[depth*nr:depth*nr+width], b[depth*n+column:depth*n+column+width])
			}
		}
	}
	return packed
}

func SGEMMPacked(c, a []float32, b *PackedB, m, maxWorkers int) {
	sgemmOpWorkers(c, a, nil, m, b.n, b.k, false, false, maxWorkers, b)
}

func sgemmBuffer(n int) []float32 {
	if n <= 0 {
		return make([]float32, 0)
	}
	if n > maxSGEMMPoolSize {
		return make([]float32, n)
	}
	idx, size := sgemmPoolBucket(n)
	if v := sgemmPools[idx].Get(); v != nil {
		return v.([]float32)[:n]
	}
	return make([]float32, n, size)
}
func releaseSGEMMBuffer(v []float32) {
	c := cap(v)
	if c == 0 || c > maxSGEMMPoolSize {
		return
	}
	idx, size := sgemmPoolBucket(c)
	if c != size {
		return
	}
	sgemmPools[idx].Put(v[:size])
}

// SGEMM computes C = A*B for contiguous row-major matrices.
func SGEMM(c, a, b []float32, m, n, k int) {
	SGEMMOp(c, a, b, m, n, k, false, false)
}

// SGEMMOp computes C = op(A)*op(B), with row-major stored operands.
// transA stores A as k-by-m; transB stores B as n-by-k.
func SGEMMOp(c, a, b []float32, m, n, k int, transA, transB bool) {
	SGEMMOpWorkers(c, a, b, m, n, k, transA, transB, runtime.GOMAXPROCS(0))
}

// SGEMMOpWorkers caps internal row parallelism, useful when independent batch
// matrices are already processed by outer workers.
func SGEMMOpWorkers(c, a, b []float32, m, n, k int, transA, transB bool, maxWorkers int) {
	sgemmOpWorkers(c, a, b, m, n, k, transA, transB, maxWorkers, nil)
}

const sgemmSmallDim = 8

func sgemmSmall(c, a, b []float32, m, n, k int, transA, transB bool) {
	aRow, aDepth := k, 1
	bDepth, bCol := n, 1
	if transA {
		aRow, aDepth = 1, m
	}
	if transB {
		bDepth, bCol = 1, k
	}
	var out [sgemmSmallDim * sgemmSmallDim]float32
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ai, bi := i*aRow, j*bCol
			var sum float32
			for d := 0; d < k; d++ {
				sum += a[ai] * b[bi]
				ai += aDepth
				bi += bDepth
			}
			out[i*n+j] = sum
		}
	}
	copy(c, out[:m*n])
}

func sgemmOpWorkers(c, a, b []float32, m, n, k int, transA, transB bool, maxWorkers int, packed *PackedB) {
	if m < 0 || n < 0 || k < 0 || len(a) != m*k || (packed == nil && len(b) != k*n) || len(c) != m*n {
		panic("tensor: SGEMM shape mismatch")
	}
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		clear(c)
		return
	}
	if packed == nil && max(m, n, k) <= sgemmSmallDim {
		sgemmSmall(c, a, b, m, n, k, transA, transB)
		return
	}
	// A short row block keeps enough tiles to spread over the workers when m is
	// small; tall matrices already have plenty.
	mc := 36
	if m <= 512 {
		mc = 18
	}
	if !use4x16 {
		mc = 32
		if m <= 512 {
			mc = 16
		}
	}
	const kc = 256
	mr, nr := sgemmTile()
	groups := (n + nr - 1) / nr
	var bp []float32
	if packed != nil {
		bp = packed.data
	} else {
		bp = sgemmBuffer(groups * k * nr)
		defer releaseSGEMMBuffer(bp)
		// The panel packing is independent per group and otherwise runs on one
		// core while the rest of the machine waits.
		pack := func(g int) {
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
		packWorkers := min(runtime.GOMAXPROCS(0), groups)
		if packWorkers > 1 && groups*k*nr >= 65536 {
			var wg sync.WaitGroup
			for w := 0; w < packWorkers; w++ {
				wg.Add(1)
				go func(start, end int) {
					defer wg.Done()
					for g := start; g < end; g++ {
						pack(g)
					}
				}(groups*w/packWorkers, groups*(w+1)/packWorkers)
			}
			wg.Wait()
		} else {
			for g := 0; g < groups; g++ {
				pack(g)
			}
		}
	}
	tiles := (m + mc - 1) / mc
	workers := min(max(1, maxWorkers), tiles)
	var jobs chan int
	run := func() {
		ap := sgemmBuffer(mc * kc)
		defer releaseSGEMMBuffer(ap)
		// 端の小さなブロックも、一時領域でAVXのカーネルを使う。余った行と列の結果は捨てる。
		var edge [96]float32
		process := func(row int) {
			end := min(row+mc, m)
			for depth := 0; depth < k; depth += kc {
				nk := min(kc, k-depth)
				for i := row; i < end; i += mr {
					height := min(mr, end-i)
					panel := ap[(i-row)*kc : (i-row+mr)*kc]
					if transA {
						for d := 0; d < nk; d++ {
							copy(panel[d*mr:d*mr+height], a[(depth+d)*m+i:(depth+d)*m+i+height])
						}
					} else {
						for r := 0; r < height; r++ {
							source := a[(i+r)*k+depth : (i+r)*k+depth+nk]
							for d, v := range source {
								panel[d*mr+r] = v
							}
						}
					}
				}
				for g := 0; g < groups; g++ {
					j := g * nr
					width := min(n-j, nr)
					bb := bp[g*k*nr+depth*nr:]
					for i := row; i < end; i += mr {
						height := min(mr, end-i)
						aa := ap[(i-row)*kc:]
						if height == mr && width == nr {
							if use4x16 {
								kernel6x16AVX(&c[i*n+j], &aa[0], &bb[0], n, nk, depth != 0)
							} else if useAVX2FMA {
								kernel4x8AVX(&c[i*n+j], &aa[0], &bb[0], n, nk, depth != 0)
							} else {
								kernel4x4(c, aa, bb, i*n+j, n, nk, depth != 0)
							}
						} else if use4x16 {
							if depth != 0 {
								for r := 0; r < height; r++ {
									copy(edge[r*nr:r*nr+width], c[(i+r)*n+j:(i+r)*n+j+width])
								}
							}
							kernel6x16AVX(&edge[0], &aa[0], &bb[0], nr, nk, depth != 0)
							for r := 0; r < height; r++ {
								copy(c[(i+r)*n+j:(i+r)*n+j+width], edge[r*nr:r*nr+width])
							}
						} else {
							for r := 0; r < height; r++ {
								for q := 0; q < width; q++ {
									var sum float32
									if depth != 0 {
										sum = c[(i+r)*n+j+q]
									}
									for d := 0; d < nk; d++ {
										sum += aa[d*mr+r] * bb[d*nr+q]
									}
									c[(i+r)*n+j+q] = sum
								}
							}
						}
					}
				}
			}
		}
		if jobs == nil {
			for row := 0; row < m; row += mc {
				process(row)
			}
		} else {
			for row := range jobs {
				process(row)
			}
		}
	}
	if workers == 1 {
		// Batched GEMMs often already have outer workers. Avoid launching an
		// extra goroutine or creating a job channel in that common case.
		run()
		return
	}
	// Keep dynamic tile assignment for multiple workers: it tolerates uneven
	// CPU availability better than assigning a fixed set of rows per worker.
	jobs = make(chan int, tiles)
	var wg sync.WaitGroup
	for w := 1; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			run()
		}()
	}
	for row := 0; row < m; row += mc {
		jobs <- row
	}
	close(jobs)
	run()
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
