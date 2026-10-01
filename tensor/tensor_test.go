package tensor

import (
	"testing"

	"github.com/yh2237/gograd/cuda"
)

func TestTensorRoundTrip(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable")
	}
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	data := []float32{1, 2, 3, 4, 5, 6}
	tr, err := FromHost([]int{2, 3}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if tr.Numel() != 6 {
		t.Fatalf("numel %d != 6", tr.Numel())
	}
	got, err := tr.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	for i := range data {
		if got[i] != data[i] {
			t.Fatalf("round trip[%d]: got %v want %v", i, got[i], data[i])
		}
	}
	view, err := tr.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if shape := view.Shape(); shape[0] != 3 || shape[1] != 2 {
		t.Fatalf("reshape shape %v", shape)
	}
	if _, err := tr.Reshape(4); err == nil {
		t.Fatal("expected reshape element mismatch")
	}

	zeros, err := Zeros(4)
	if err != nil {
		t.Fatal(err)
	}
	defer zeros.Close()
	values, err := zeros.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range values {
		if value != 0 {
			t.Fatalf("zeros[%d] = %v", i, value)
		}
	}
}
