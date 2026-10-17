package autograd

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

func naiveConv1d(x, w, b, upstream []float32, batch, length, ci, co, kernel, stride, padding, dilation int) (y, dx, dw, db []float32) {
	outLength := (length+2*padding-dilation*(kernel-1)-1)/stride + 1
	y = make([]float32, batch*outLength*co)
	dx, dw, db = make([]float32, len(x)), make([]float32, len(w)), make([]float32, len(b))
	for n := 0; n < batch; n++ {
		for at := 0; at < outLength; at++ {
			for o := 0; o < co; o++ {
				index := (n*outLength+at)*co + o
				y[index] = b[o]
				if upstream != nil {
					db[o] += upstream[index]
				}
				for c := 0; c < ci; c++ {
					for tap := 0; tap < kernel; tap++ {
						src := at*stride - padding + tap*dilation
						if src < 0 || src >= length {
							continue
						}
						xi, wi := (n*length+src)*ci+c, (o*ci+c)*kernel+tap
						y[index] += x[xi] * w[wi]
						if upstream != nil {
							dx[xi] += upstream[index] * w[wi]
							dw[wi] += upstream[index] * x[xi]
						}
					}
				}
			}
		}
	}
	return
}

func TestConv1dGEMMStrideReference(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA {
				if !cuda.Available() {
					t.Skip("CUDA unavailable")
				}
			}
			for _, tc := range []struct {
				name                                      string
				length, kernel, stride, padding, dilation int
			}{
				{"hubert10", 27, 10, 5, 0, 1},
				{"hubert2", 13, 2, 2, 0, 1},
				{"hubert3", 14, 3, 2, 0, 1},
				{"padded_dilated", 12, 4, 2, 2, 2},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if device == tensor.CUDA {
						ctx, err := NewCUDAContext()
						if err != nil {
							t.Fatal(err)
						}
						defer ctx.Close()
					}
					const batch, ci, co = 2, 3, 4
					xv := make([]float32, batch*tc.length*ci)
					wv := make([]float32, co*ci*tc.kernel)
					bv := make([]float32, co)
					for i := range xv {
						xv[i] = float32(math.Sin(float64(i+1)*.37)) * .3
					}
					for i := range wv {
						wv[i] = float32(math.Cos(float64(i+1)*.23)) * .2
					}
					for i := range bv {
						bv[i] = float32(i-2) * .1
					}
					outLength := (tc.length+2*tc.padding-tc.dilation*(tc.kernel-1)-1)/tc.stride + 1
					up := make([]float32, batch*outLength*co)
					for i := range up {
						up[i] = float32(math.Sin(float64(i+2)*.19)) * .4
					}
					wantY, wantX, wantW, wantB := naiveConv1d(xv, wv, bv, up, batch, tc.length, ci, co, tc.kernel, tc.stride, tc.padding, tc.dilation)
					x := conv2dTestTensor(t, xv, []int{batch, tc.length, ci}, device, true)
					w := conv2dTestTensor(t, wv, []int{co, ci, tc.kernel}, device, true)
					b := conv2dTestTensor(t, bv, []int{co}, device, true)
					y := Conv1dGEMMStride(x, w, b, tc.stride, tc.padding, tc.dilation)
					if y.Shape[1] != outLength {
						t.Fatalf("output length %d want %d", y.Shape[1], outLength)
					}
					conv2dClose(t, "output", conv2dHost(t, y, false), wantY, 2e-5, 2e-5)
					loss := Sum(Mul(y, conv2dTestTensor(t, up, y.Shape, device, false)), 0, 1, 2)
					if err := loss.Backward(); err != nil {
						t.Fatal(err)
					}
					conv2dClose(t, "input grad", conv2dHost(t, x, true), wantX, 3e-5, 3e-5)
					conv2dClose(t, "weight grad", conv2dHost(t, w, true), wantW, 3e-5, 3e-5)
					conv2dClose(t, "bias grad", conv2dHost(t, b, true), wantB, 3e-5, 3e-5)
					loss.ReleaseGraph()
				})
			}
		})
	}
}

func TestConv1dGEMMStrideFiniteDifferences(t *testing.T) {
	x := Must([]float32{.2, -.1, .3, .4, -.2, .1, .5}, []int{1, 7, 1}, true)
	w := Must([]float32{.1, -.3, .2, .4}, []int{1, 1, 4}, true)
	b := Must([]float32{.05}, []int{1}, true)
	up := Must([]float32{.3, -.2, .4}, []int{1, 3, 1}, false)
	loss := Sum(Mul(Conv1dGEMMStride(x, w, b, 2, 1, 1), up), 0, 1, 2)
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	loss.ReleaseGraph()
	for name, p := range map[string]*Tensor{"input": x, "weight": w, "bias": b} {
		for i, original := range p.Data {
			const eps float32 = .001
			p.Data[i] = original + eps
			plus := conv1dStrideLoss(x, w, b, up)
			p.Data[i] = original - eps
			minus := conv1dStrideLoss(x, w, b, up)
			p.Data[i] = original
			want := (plus - minus) / float64(2*eps)
			if math.Abs(float64(p.Grad[i])-want) > 2e-4 {
				t.Fatalf("%s[%d] analytic=%g numerical=%g", name, i, p.Grad[i], want)
			}
		}
	}
}

func conv1dStrideLoss(x, w, b, up *Tensor) (sum float64) {
	NoGrad(func() {
		y := Conv1dGEMMStride(x, w, b, 2, 1, 1)
		defer y.ReleaseGraph()
		for i, v := range y.Data {
			sum += float64(v) * float64(up.Data[i])
		}
	})
	return
}
