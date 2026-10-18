package vulkan

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

func TestVulkanGELU(t *testing.T) {
	b := testBackend(t)
	for _, n := range []int{0, 1, 1031} {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i%251-125) / 16
		}
		if n > 10 {
			copy(data, []float32{-1e20, -100, -10, -5, -1, 0, 1, 5, 10, 100, 1e20})
		}
		x, out := testUpload(t, b, data), testAlloc(t, b, n)
		for _, approx := range []bool{false, true} {
			want := make([]float32, n)
			for i, value := range data {
				v := float64(value)
				if approx {
					want[i] = float32(.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+.044715*v*v*v))))
				} else {
					want[i] = float32(.5 * v * (1 + math.Erf(v/math.Sqrt2)))
				}
			}
			var err error
			if approx {
				err = b.GELU(out, x, n)
			} else {
				err = b.GELUExact(out, x, n)
			}
			if err != nil {
				t.Fatal(err)
			}
			checkMatrix(t, b, out, want)
		}
	}
}

func TestVulkanSoftmax(t *testing.T) {
	b := testBackend(t)
	for _, cols := range []int{1, 7, 129, 768, 3072} {
		t.Run(fmt.Sprint(cols), func(t *testing.T) {
			const rows = 9
			data, want := make([]float32, rows*cols), make([]float32, rows*cols)
			for row := 0; row < rows; row++ {
				mx := float32(math.Inf(-1))
				for j := 0; j < cols; j++ {
					v := float32((row*31+j*13)%101-50)/8 + float32(row-4)*1000
					if j%11 == 1 {
						v = float32(math.Inf(-1))
					}
					data[row*cols+j] = v
					mx = max(mx, v)
				}
				var sum float32
				for j := 0; j < cols; j++ {
					want[row*cols+j] = float32(math.Exp(float64(data[row*cols+j] - mx)))
					sum += want[row*cols+j]
				}
				for j := 0; j < cols; j++ {
					want[row*cols+j] /= sum
				}
			}
			out := testAlloc(t, b, len(data))
			if err := b.Softmax(out, testUpload(t, b, data), rows, cols); err != nil {
				t.Fatal(err)
			}
			checkMatrix(t, b, out, want)
			got, err := b.Download(out, len(data))
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < rows; row++ {
				var sum float64
				for _, v := range got[row*cols : (row+1)*cols] {
					sum += float64(v)
				}
				if math.Abs(sum-1) > 1e-5 {
					t.Fatalf("row sum %g", sum)
				}
			}
		})
	}
	if err := b.Softmax(testAlloc(t, b, 0), testAlloc(t, b, 0), 0, 17); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanNormalization(t *testing.T) {
	b := testBackend(t)
	for _, cols := range []int{1, 7, 129, 768, 3072} {
		t.Run(fmt.Sprintf("layer/%d", cols), func(t *testing.T) {
			const rows = 5
			data, want := make([]float32, rows*cols), make([]float32, rows*cols)
			w, bias := make([]float32, cols), make([]float32, cols)
			for j := range w {
				w[j] = float32(j%7-3) / 5
				bias[j] = float32(j%11-5) / 8
			}
			for i := range data {
				data[i] = float32(i%37-18) / 8
			}
			for j := 0; j < cols; j++ {
				data[j] = 3
			} // zero variance
			for row := 0; row < rows; row++ {
				var mean, variance float32
				for _, v := range data[row*cols : (row+1)*cols] {
					mean += v
				}
				mean /= float32(cols)
				for _, v := range data[row*cols : (row+1)*cols] {
					z := v - mean
					variance += z * z
				}
				inv := float32(1 / math.Sqrt(float64(variance/float32(cols)+1e-5)))
				for j := 0; j < cols; j++ {
					want[row*cols+j] = (data[row*cols+j]-mean)*inv*w[j] + bias[j]
				}
			}
			out := testAlloc(t, b, len(data))
			if err := b.LayerNorm(out, testUpload(t, b, data), testUpload(t, b, w), testUpload(t, b, bias), rows, cols, 1e-5); err != nil {
				t.Fatal(err)
			}
			checkMatrix(t, b, out, want)
		})
	}
	for _, dims := range [][4]int{{2, 1, 6, 3}, {2, 17, 12, 3}, {2, 129, 8, 1}, {1, 3, 8, 8}} {
		t.Run(fmt.Sprintf("group/%v", dims), func(t *testing.T) {
			bs, frames, channels, groups := dims[0], dims[1], dims[2], dims[3]
			cg := channels / groups
			data, want := make([]float32, bs*frames*channels), make([]float32, bs*frames*channels)
			w, bias := make([]float32, channels), make([]float32, channels)
			for i := range data {
				data[i] = float32(i%37-18) / 8
			}
			for i := range w {
				w[i] = float32(i%7-3) / 5
				bias[i] = float32(i%11-5) / 8
			}
			for n := 0; n < bs; n++ {
				for gr := 0; gr < groups; gr++ {
					var mean, variance float32
					for at := 0; at < frames; at++ {
						for c := gr * cg; c < (gr+1)*cg; c++ {
							mean += data[(n*frames+at)*channels+c]
						}
					}
					mean /= float32(frames * cg)
					for at := 0; at < frames; at++ {
						for c := gr * cg; c < (gr+1)*cg; c++ {
							z := data[(n*frames+at)*channels+c] - mean
							variance += z * z
						}
					}
					inv := float32(1 / math.Sqrt(float64(variance/float32(frames*cg)+1e-5)))
					for at := 0; at < frames; at++ {
						for c := gr * cg; c < (gr+1)*cg; c++ {
							i := (n*frames+at)*channels + c
							want[i] = (data[i]-mean)*inv*w[c] + bias[c]
						}
					}
				}
			}
			out := testAlloc(t, b, len(data))
			if err := b.GroupNorm(out, testUpload(t, b, data), testUpload(t, b, w), testUpload(t, b, bias), bs, frames, channels, groups, 1e-5); err != nil {
				t.Fatal(err)
			}
			checkMatrix(t, b, out, want)
		})
	}
	for _, group := range []bool{false, true} {
		out, x, w, bias := testAlloc(t, b, 0), testAlloc(t, b, 0), testAlloc(t, b, 4), testAlloc(t, b, 4)
		var err error
		if group {
			err = b.GroupNorm(out, x, w, bias, 0, 3, 4, 2, 1e-5)
		} else {
			err = b.LayerNorm(out, x, w, bias, 0, 4, 1e-5)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestVulkanConv1d(t *testing.T) {
	b := testBackend(t)
	for _, o := range []gpu.Conv1dOptions{
		{Batch: 2, InputLength: 17, InChannels: 3, OutChannels: 7, Kernel: 3, Stride: 1, Padding: 1, Dilation: 1},
		{Batch: 2, InputLength: 19, InChannels: 3, OutChannels: 5, Kernel: 4, Stride: 2, Padding: 3, Dilation: 2},
		{Batch: 1, InputLength: 3, InChannels: 1, OutChannels: 1, Kernel: 1, Stride: 9, Padding: 7, Dilation: 1},
	} {
		for _, hasBias := range []bool{false, true} {
			t.Run(fmt.Sprintf("%+v/bias=%t", o, hasBias), func(t *testing.T) {
				length, err := o.OutputLength()
				if err != nil {
					t.Fatal(err)
				}
				x, w, bias := make([]float32, o.Batch*o.InputLength*o.InChannels), make([]float32, o.OutChannels*o.InChannels*o.Kernel), make([]float32, o.OutChannels)
				for i := range x {
					x[i] = float32(i%37-18) / 8
				}
				for i := range w {
					w[i] = float32(i%17-8) / 32
				}
				for i := range bias {
					bias[i] = float32(i%7-3) / 5
				}
				want := make([]float32, o.Batch*length*o.OutChannels)
				for n := 0; n < o.Batch; n++ {
					for at := 0; at < length; at++ {
						for co := 0; co < o.OutChannels; co++ {
							var sum float32
							for ci := 0; ci < o.InChannels; ci++ {
								for k := 0; k < o.Kernel; k++ {
									src := at*o.Stride - o.Padding + k*o.Dilation
									if src >= 0 && src < o.InputLength {
										sum += x[(n*o.InputLength+src)*o.InChannels+ci] * w[(co*o.InChannels+ci)*o.Kernel+k]
									}
								}
							}
							if hasBias {
								sum += bias[co]
							}
							want[(n*length+at)*o.OutChannels+co] = sum
						}
					}
				}
				var v *gpu.Buffer
				if hasBias {
					v = testUpload(t, b, bias)
				}
				out := testAlloc(t, b, len(want))
				if err := b.Conv1d(out, testUpload(t, b, x), testUpload(t, b, w), v, o); err != nil {
					t.Fatal(err)
				}
				checkMatrix(t, b, out, want)
			})
		}
	}
}

func TestVulkanTransformerValidation(t *testing.T) {
	b := testBackend(t)
	x, y, w, bias := testAlloc(t, b, 8), testAlloc(t, b, 8), testAlloc(t, b, 4), testAlloc(t, b, 4)
	for _, eps := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if err := b.LayerNorm(y, x, w, bias, 2, 4, eps); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
		if err := b.GroupNorm(y, x, w, bias, 1, 2, 4, 2, eps); !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
	}
	for _, err := range []error{b.Softmax(y, x, 2, 0), b.Softmax(x, x, 2, 4), b.GELU(x, x, 8), b.LayerNorm(x, x, w, bias, 2, 4, 1e-5), b.GroupNorm(y, x, w, bias, 1, 2, 4, 3, 1e-5), b.LayerNorm(y, x, w, w, 1, 8, 1e-5)} {
		if !errors.Is(err, gpu.ErrInvalidArgument) {
			t.Fatal(err)
		}
	}
	if !b.Available() {
		t.Fatal("validation disabled healthy device")
	}
}
