package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func TestViewsAndNDBroadcastReduction(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
				ctx, err := NewCUDAContext()
				if err != nil {
					t.Skip(err)
				}
				defer ctx.Close()
			}
			xdata := make([]float32, 2*3*2*2*2*2)
			for i := range xdata {
				xdata[i] = float32(i%13) / 7
			}
			x, err := New(xdata, []int{2, 3, 2, 2, 2, 2}, device, true)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			b, err := New([]float32{2, 3, 4}, []int{1, 3, 1, 1, 1, 1}, device, true)
			if err != nil {
				t.Fatal(err)
			}
			defer b.Close()
			p := Permute(x, 1, 0, 2, 3, 4, 5)
			s := Slice(p, 0, 1, 3)
			if s.IsContiguous() {
				t.Fatal("slice unexpectedly contiguous")
			}
			v, err := s.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(v[0]-xdata[16])) > 1e-6 {
				t.Fatalf("view value %v", v[0])
			}
			viewLoss := Sum(s, 0, 1, 2, 3, 4, 5)
			if err := viewLoss.Backward(); err != nil {
				t.Fatal(err)
			}
			viewGrad, err := x.GradToHost()
			if err != nil {
				t.Fatal(err)
			}
			for i, got := range viewGrad {
				want := float32(0)
				if (i/16)%3 != 0 {
					want = 1
				}
				if math.Abs(float64(got-want)) > 1e-5 {
					t.Fatalf("view grad[%d]=%v want %v", i, got, want)
				}
			}
			x.ZeroGrad()
			expandBase, err := New([]float32{1, 2}, []int{1, 2, 1, 1, 1, 1}, device, true)
			if err != nil {
				t.Fatal(err)
			}
			if err := Sum(Expand(expandBase, 2, 2, 2, 2, 2, 2), 0, 1, 2, 3, 4, 5).Backward(); err != nil {
				t.Fatal(err)
			}
			expandGrad, err := expandBase.GradToHost()
			if err != nil {
				t.Fatal(err)
			}
			for i, g := range expandGrad {
				if math.Abs(float64(g-32)) > 1e-5 {
					t.Fatalf("expand grad[%d]=%v", i, g)
				}
			}
			expandBase.Close()
			y := Mul(x, b)
			z := Mean(y, 0, 2, 3, 4, 5)
			got, err := z.ToHost()
			if err != nil {
				t.Fatal(err)
			}
			for j := 0; j < 3; j++ {
				want := float32(0)
				for i := 0; i < 2; i++ {
					for k := 0; k < 16; k++ {
						want += xdata[(i*3+j)*16+k] * float32(2+j) / 32
					}
				}
				if math.Abs(float64(got[j]-want)) > 1e-4 {
					t.Fatalf("mean[%d]=%v want %v", j, got[j], want)
				}
			}
			Sum(z, 0).Backward()
			g, err := x.GradToHost()
			if err != nil {
				t.Fatal(err)
			}
			for i := range g {
				if math.Abs(float64(g[i]-float32(2+i/16%3)/32)) > 1e-5 {
					t.Fatalf("grad[%d]=%v", i, g[i])
				}
			}
			z.ReleaseGraph()
		})
	}
}
