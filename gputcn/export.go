package gputcn

import (
	"math"

	"github.com/yh2237/gograd"
)

func to64(values []float32) []float64 {
	result := make([]float64, len(values))
	for i, value := range values {
		result[i] = float64(value)
	}
	return result
}

func matrix(values []float32, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = to64(values[r*cols : (r+1)*cols])
	}
	return result
}

func tensor3(values []float32, outer, middle, inner int) [][][]float64 {
	result := make([][][]float64, outer)
	for o := 0; o < outer; o++ {
		result[o] = make([][]float64, middle)
		for m := 0; m < middle; m++ {
			base := (o*middle + m) * inner
			result[o][m] = to64(values[base : base+inner])
		}
	}
	return result
}

// ExportFramePitch downloads the device parameters and builds the runtime JSON
// layout. The output scale is folded into the output weight and bias.
func (m *Model) ExportFramePitch(featureNames []string, frameMS, lowCents, highCents, targetScale float64) (*gograd.FramePitchExport, error) {
	scale := math.Max(1, targetScale)
	inputWeight, err := DownloadFloat32(m.InputWeight, m.Hidden*m.Inputs)
	if err != nil {
		return nil, err
	}
	inputBias, err := DownloadFloat32(m.InputBias, m.Hidden)
	if err != nil {
		return nil, err
	}
	outputWeight, err := DownloadFloat32(m.OutputWeight, m.Hidden)
	if err != nil {
		return nil, err
	}
	outputBias, err := DownloadFloat32(m.OutputBias, 1)
	if err != nil {
		return nil, err
	}
	export := &gograd.FramePitchExport{
		FeatureNames:     append([]string(nil), featureNames...),
		InputWeights:     matrix(inputWeight, m.Hidden, m.Inputs),
		InputBias:        to64(inputBias),
		OutputWeight:     make([]float64, m.Hidden),
		OutputBias:       float64(outputBias[0]) * scale,
		FrameMS:          frameMS,
		LowCents:         lowCents,
		HighCents:        highCents,
		Centered:         true,
		TargetScaleCents: scale,
	}
	for i := range export.OutputWeight {
		export.OutputWeight[i] = float64(outputWeight[i]) * scale
	}
	for i := range m.Layers {
		weight, err := DownloadFloat32(m.Layers[i].Weight, m.Hidden*m.Hidden*3)
		if err != nil {
			return nil, err
		}
		bias, err := DownloadFloat32(m.Layers[i].Bias, m.Hidden)
		if err != nil {
			return nil, err
		}
		export.Layers = append(export.Layers, gograd.LayerExport{
			Dilation: m.Layers[i].Dilation,
			Weights:  tensor3(weight, m.Hidden, m.Hidden, 3),
			Bias:     to64(bias),
		})
	}
	return export, nil
}
