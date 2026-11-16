package kernels

import (
	"math"
	"runtime"
	"testing"

	"github.com/yh2237/gograd/cuda"
)

func geluTanh(v float32) float32 {
	u := 0.7978845608028654 * (float64(v) + 0.044715*float64(v)*float64(v)*float64(v))
	return float32(0.5 * float64(v) * (1 + math.Tanh(u)))
}

func geluTanhGrad(v float32) float32 {
	x := float64(v)
	u := 0.7978845608028654 * (x + 0.044715*x*x*x)
	th := math.Tanh(u)
	return float32(0.5*(1+th) + 0.5*x*(1-th*th)*0.7978845608028654*(1+3*0.044715*x*x))
}

func TestActivationKernels(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	input := []float32{-2, -0.5, 0, 0.5, 2}
	upstream := []float32{1, -2, 3, -4, 5}
	const slope float32 = 0.1
	deviceInput := upload(t, input)
	defer deviceInput.Free()
	deviceGrad := upload(t, upstream)
	defer deviceGrad.Free()
	out, err := cuda.Alloc(len(input) * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	for _, tc := range []struct {
		name     string
		kind     int
		forward  func(float32) float32
		backward func(float32) float32
	}{
		{"relu", 0,
			func(v float32) float32 { return max(v, 0) },
			func(v float32) float32 {
				if v > 0 {
					return 1
				}
				return 0
			}},
		{"leaky_relu", 1,
			func(v float32) float32 {
				if v > 0 {
					return v
				}
				return slope * v
			},
			func(v float32) float32 {
				if v > 0 {
					return 1
				}
				return slope
			}},
		{"gelu", 2, geluTanh, geluTanhGrad},
	} {
		if err := Activation(deviceInput, nil, out, len(input), tc.kind, slope, false); err != nil {
			t.Fatalf("%s forward: %v", tc.name, err)
		}
		got := download(t, out, len(input))
		for i, v := range input {
			if want := tc.forward(v); math.Abs(float64(got[i]-want)) > 1e-5 {
				t.Fatalf("%s forward[%d] = %g, want %g", tc.name, i, got[i], want)
			}
		}
		if err := Activation(deviceInput, deviceGrad, out, len(input), tc.kind, slope, true); err != nil {
			t.Fatalf("%s backward: %v", tc.name, err)
		}
		got = download(t, out, len(input))
		for i, v := range input {
			want := upstream[i] * tc.backward(v)
			if math.Abs(float64(got[i]-want)) > 1e-5 {
				t.Fatalf("%s backward[%d] = %g, want %g", tc.name, i, got[i], want)
			}
		}
	}
}

func TestActivationEmpty(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	empty, err := cuda.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Free()
	if err := Activation(empty, nil, empty, 0, 0, 0, false); err != nil {
		t.Fatalf("empty activation: %v", err)
	}
}
