package irodori

import (
	"encoding/json"
	"os"
	"testing"
)

type convCase struct {
	In     int   `json:"in"`
	Out    int   `json:"out"`
	Kernel int   `json:"kernel"`
	Bins   int   `json:"bins"`
	Frames int   `json:"frames"`
	Offset int64 `json:"offset"`
	Bytes  int   `json:"bytes"`
}

type convFixture struct {
	Cases []convCase `json:"cases"`
}

// loadConvFixture reads the reference's Layer. Each case packs the input grid,
// the eight parameter tensors, and then the plain, gated and normalized
// outputs, all little endian float32.
func loadConvFixture(t *testing.T) ([]convCase, []byte) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/irodori_conv2d.json")
	if err != nil {
		t.Skip(err)
	}
	var f convFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile("../testdata/irodori_conv2d.bin")
	if err != nil {
		t.Fatal(err)
	}
	return f.Cases, blob
}

// TestConv2dMatchesTorch checks the shifted-product convolution against
// nn.Conv2d for the channel counts and kernel sizes the watermark uses.
func TestConv2dMatchesTorch(t *testing.T) {
	cases, blob := loadConvFixture(t)
	for _, c := range cases {
		at := c.Offset
		input := readF32(blob, at, c.In*c.Bins*c.Frames)
		at += int64(c.In*c.Bins*c.Frames) * 4
		reads := map[string][]float32{}
		shapes := map[string][]int{
			"convWeight": {c.Out, c.In, c.Kernel, c.Kernel}, "convBias": {c.Out},
			"gateWeight": {c.Out, c.In, c.Kernel, c.Kernel}, "gateBias": {c.Out},
			"gamma": {c.Out}, "beta": {c.Out},
		}
		for _, name := range []string{"convWeight", "convBias", "gateWeight", "gateBias", "gamma", "beta"} {
			n := 1
			for _, d := range shapes[name] {
				n *= d
			}
			reads[name] = readF32(blob, at, n)
			at += int64(n) * 4
		}
		outElems := c.Out * c.Bins * c.Frames
		wantPlain := readF32(blob, at, outElems)
		at += int64(outElems) * 4
		wantGated := readF32(blob, at, outElems)
		at += int64(outElems) * 4
		wantNormed := readF32(blob, at, outElems)

		ws := &workspace{}
		conv := conv2d{weight: reads["convWeight"], bias: reads["convBias"],
			inChannels: c.In, outChans: c.Out, kernel: c.Kernel}
		gotPlain, err := conv.forward(grid{data: input, channels: c.In, bins: c.Bins, rows: c.Frames}, ws)
		if err != nil {
			t.Fatal(err)
		}
		abs, at2 := maxAbs32(gotPlain.data, wantPlain)
		scale := peakOf(wantPlain)
		t.Logf("%d->%d k=%d plain max_abs=%.9g (scale %.9g, at %d)", c.In, c.Out, c.Kernel, abs, scale, at2)
		if abs > 1e-4*float64(scale)+1e-5 {
			t.Errorf("%d->%d k=%d plain convolution differs by %.9g at %d", c.In, c.Out, c.Kernel, abs, at2)
		}

		gate := conv2d{weight: reads["gateWeight"], bias: reads["gateBias"],
			inChannels: c.In, outChans: c.Out, kernel: c.Kernel}
		gotGated, err := gate.forward(grid{data: input, channels: c.In, bins: c.Bins, rows: c.Frames}, ws)
		if err != nil {
			t.Fatal(err)
		}
		// The reference multiplies the main convolution by a sigmoid of the
		// gate convolution, so the product is what the fixture records.
		for i := range gotPlain.data {
			gotPlain.data[i] *= sigmoid(gotGated.data[i])
		}
		abs, at2 = maxAbs32(gotPlain.data, wantGated)
		scale = peakOf(wantGated)
		t.Logf("%d->%d k=%d gated max_abs=%.9g (scale %.9g)", c.In, c.Out, c.Kernel, abs, scale)
		if abs > 1e-4*float64(scale)+1e-5 {
			t.Errorf("%d->%d k=%d gated convolution differs by %.9g at %d", c.In, c.Out, c.Kernel, abs, at2)
		}

		// A whole layer: gated convolution then batch normalization over the
		// input's own statistics, which is how the reference runs it.
		l := layer{conv: conv, gate: gate, gamma: reads["gamma"], beta: reads["beta"]}
		gotNormed, err := l.forward(grid{data: input, channels: c.In, bins: c.Bins, rows: c.Frames}, ws)
		if err != nil {
			t.Fatal(err)
		}
		abs, at2 = maxAbs32(gotNormed.data, wantNormed)
		normedScale := peakOf(wantNormed)
		t.Logf("%d->%d k=%d layer max_abs=%.9g (scale %.9g)", c.In, c.Out, c.Kernel, abs, normedScale)
		if abs > 1e-4*float64(normedScale)+1e-5 {
			t.Errorf("%d->%d k=%d layer differs by %.9g at %d", c.In, c.Out, c.Kernel, abs, at2)
		}
	}
}
