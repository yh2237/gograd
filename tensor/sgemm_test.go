package tensor

import "testing"

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
