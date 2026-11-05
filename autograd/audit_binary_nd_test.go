package autograd

import "testing"

// The rank-four elementwise kernel has two index paths: equal shapes use the
// identity fast path, while a broadcast operand still uses the strided index.
func TestAuditBinaryRankFourBroadcast(t *testing.T) {
	a := Must([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 1, 2}, true)
	b := Must([]float32{2, 3, 4, 5, 6, 7, 8, 9}, []int{2, 2, 1, 2}, true)
	loss := Sum(Mul(a, b), 0, 1, 2, 3)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	closeValues(t, "equal a", a.Grad, []float32{2, 3, 4, 5, 6, 7, 8, 9}, 0)
	closeValues(t, "equal b", b.Grad, []float32{1, 2, 3, 4, 5, 6, 7, 8}, 0)
	loss.ReleaseGraph()

	x := Must([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 1, 2}, true)
	w := Must([]float32{2, 3}, []int{1, 1, 1, 2}, true)
	loss = Sum(Mul(x, w), 0, 1, 2, 3)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	closeValues(t, "broadcast x", x.Grad, []float32{2, 3, 2, 3, 2, 3, 2, 3}, 0)
	closeValues(t, "broadcast w", w.Grad, []float32{16, 20}, 0)
	loss.ReleaseGraph()
}
