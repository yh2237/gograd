package tensor

import (
	"fmt"
	"strings"
	"testing"
)

func naive(c, a, b []float32, m, n, k int) {
	clear(c)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			for d := 0; d < k; d++ {
				c[i*n+j] += a[i*k+d] * b[d*n+j]
			}
		}
	}
}
func TestSGEMM(t *testing.T) {
	m, n, k := 37, 53, 41
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%17-8) * .01
	}
	for i := range b {
		b[i] = float32(i%13-6) * .02
	}
	want := make([]float32, m*n)
	got := make([]float32, m*n)
	naive(want, a, b, m, n, k)
	SGEMM(got, a, b, m, n, k)
	for i := range got {
		d := got[i] - want[i]
		if d < -0.0001 || d > .0001 {
			t.Fatalf("entry %d: %g != %g", i, got[i], want[i])
		}
	}
}

func TestSGEMMEdges(t *testing.T) {
	for _, shape := range [][3]int{{0, 7, 3}, {7, 0, 3}, {7, 9, 0}, {1, 1, 1}, {5, 7, 3}, {33, 11, 257}, {67, 17, 319}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, bb, got, want := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n), make([]float32, m*n)
		for i := range a {
			a[i] = float32(i%17-8) * .01
		}
		for i := range bb {
			bb[i] = float32(i%13-6) * .02
		}
		for i := range got {
			got[i] = 1
		}
		naive(want, a, bb, m, n, k)
		SGEMM(got, a, bb, m, n, k)
		for i := range got {
			d := got[i] - want[i]
			if d < -.0001 || d > .0001 {
				t.Fatalf("shape %v entry %d: %g != %g", shape, i, got[i], want[i])
			}
		}
	}
}

func TestSGEMMOp(t *testing.T) {
	for _, shape := range [][3]int{{5, 7, 3}, {33, 11, 257}, {67, 17, 319}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, bb := make([]float32, m*k), make([]float32, k*n)
		for i := range a {
			a[i] = float32(i%17-8) * .01
		}
		for i := range bb {
			bb[i] = float32(i%13-6) * .02
		}
		at, bt := make([]float32, len(a)), make([]float32, len(bb))
		for i := 0; i < m; i++ {
			for d := 0; d < k; d++ {
				at[d*m+i] = a[i*k+d]
			}
		}
		for d := 0; d < k; d++ {
			for j := 0; j < n; j++ {
				bt[j*k+d] = bb[d*n+j]
			}
		}
		want, got := make([]float32, m*n), make([]float32, m*n)
		naive(want, a, bb, m, n, k)
		for _, tc := range []struct {
			name   string
			a, bb  []float32
			ta, tb bool
		}{
			{"NN", a, bb, false, false}, {"NT", a, bt, false, true}, {"TN", at, bb, true, false}, {"TT", at, bt, true, true},
		} {
			SGEMMOp(got, tc.a, tc.bb, m, n, k, tc.ta, tc.tb)
			for i := range got {
				d := got[i] - want[i]
				if d < -.0001 || d > .0001 {
					t.Fatalf("%s shape %v entry %d: %g != %g", tc.name, shape, i, got[i], want[i])
				}
			}
		}
	}
}
func BenchmarkSGEMM(b *testing.B) {
	m, n, k := 3200, 128, 640
	a := make([]float32, m*k)
	bb := make([]float32, k*n)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i%17) * .01
	}
	for i := range bb {
		bb[i] = float32(i%13) * .01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SGEMM(c, a, bb, m, n, k)
	}
}
func BenchmarkNaiveSGEMM(b *testing.B) {
	m, n, k := 3200, 128, 640
	a := make([]float32, m*k)
	bb := make([]float32, k*n)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i%17) * .01
	}
	for i := range bb {
		bb[i] = float32(i%13) * .01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		naive(c, a, bb, m, n, k)
	}
}

// Shapes from cmd/nn-conv-train at batch=32, time=100. The linear weight
// gradients currently use a direct accumulation loop rather than SGEMM.
func BenchmarkSGEMMTrainingShapes(b *testing.B) {
	shapes := []struct {
		name    string
		m, n, k int
	}{
		{"h128k5/conv_forward", 3200, 128, 640},
		{"h128k5/conv_grad_input", 3200, 640, 128},
		{"h128k5/conv_grad_weight", 640, 128, 3200},
		{"h128k5/input_forward", 3200, 128, 74},
		{"h128k5/input_grad_input", 3200, 74, 128},
		{"h128k5/output_forward", 3200, 72, 128},
		{"h128k5/output_grad_input", 3200, 128, 72},
		{"h64k3/conv_forward", 3200, 64, 192},
		{"h64k3/conv_grad_input", 3200, 192, 64},
		{"h64k3/conv_grad_weight", 192, 64, 3200},
		{"h64k3/input_forward", 3200, 64, 74},
		{"h64k3/input_grad_input", 3200, 74, 64},
		{"h64k3/output_forward", 3200, 72, 64},
		{"h64k3/output_grad_input", 3200, 64, 72},
	}
	for _, shape := range shapes {
		transA := strings.HasSuffix(shape.name, "conv_grad_weight")
		transB := strings.HasSuffix(shape.name, "forward")
		mode := "NN"
		if transA {
			mode = "TN"
		} else if transB {
			mode = "NT"
		}
		b.Run(fmt.Sprintf("%s/%s/M%d_N%d_K%d", shape.name, mode, shape.m, shape.n, shape.k), func(b *testing.B) {
			a := make([]float32, shape.m*shape.k)
			bb := make([]float32, shape.k*shape.n)
			c := make([]float32, shape.m*shape.n)
			for i := range a {
				a[i] = float32(i%17-8) * .01
			}
			for i := range bb {
				bb[i] = float32(i%13-6) * .02
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				SGEMMOp(c, a, bb, shape.m, shape.n, shape.k, transA, transB)
			}
			b.ReportMetric(2*float64(shape.m)*float64(shape.n)*float64(shape.k)/b.Elapsed().Seconds()/1e9*float64(b.N), "GFLOP/s")
		})
	}
}
