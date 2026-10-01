package gograd

// FramePitchExport is the intonation JSON layout: feature names, the input and
// layer parameters, and the bounded frame settings.
type FramePitchExport struct {
	FeatureNames     []string      `json:"feature_names"`
	InputWeights     [][]float64   `json:"input_weights"`
	InputBias        []float64     `json:"input_bias"`
	Layers           []LayerExport `json:"layers"`
	OutputWeight     []float64     `json:"output_weight"`
	OutputBias       float64       `json:"output_bias"`
	FrameMS          float64       `json:"frame_ms"`
	LowCents         float64       `json:"low_cents"`
	HighCents        float64       `json:"high_cents"`
	Centered         bool          `json:"centered"`
	TargetScaleCents float64       `json:"target_scale_cents"`
}

// LayerExport is one dilated convolution in FramePitchExport.
type LayerExport struct {
	Dilation int           `json:"dilation"`
	Weights  [][][]float64 `json:"weights"`
	Bias     []float64     `json:"bias"`
}

// ExportFramePitch converts a trained model into the runtime JSON layout. The
// output scale is folded into the output weight and bias.
func (m *FrameIntonationTCN) ExportFramePitch(featureNames []string, frameMS, lowCents, highCents, targetScale float64) *FramePitchExport {
	scale := targetScale
	if scale < 1 {
		scale = 1
	}
	export := &FramePitchExport{
		FeatureNames:     append([]string(nil), featureNames...),
		InputWeights:     matrix2D(m.InputWeight, m.InputWeight.Shape[0], m.InputWeight.Shape[1]),
		InputBias:        append([]float64(nil), m.InputBias.Data...),
		OutputWeight:     make([]float64, len(m.OutputWeight.Data)),
		OutputBias:       m.OutputBias.Data[0] * scale,
		FrameMS:          frameMS,
		LowCents:         lowCents,
		HighCents:        highCents,
		Centered:         true,
		TargetScaleCents: scale,
	}
	for i := range m.OutputWeight.Data {
		export.OutputWeight[i] = m.OutputWeight.Data[i] * scale
	}
	for i := range m.Layers {
		layer := &m.Layers[i]
		export.Layers = append(export.Layers, LayerExport{
			Dilation: layer.Dilation,
			Weights:  tensor3D(layer.Weight),
			Bias:     append([]float64(nil), layer.Bias.Data...),
		})
	}
	return export
}

func matrix2D(t *Tensor, rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		out[r] = append([]float64(nil), t.Data[r*cols:(r+1)*cols]...)
	}
	return out
}

func tensor3D(t *Tensor) [][][]float64 {
	outer, middle, inner := t.Shape[0], t.Shape[1], t.Shape[2]
	out := make([][][]float64, outer)
	for o := 0; o < outer; o++ {
		out[o] = make([][]float64, middle)
		for m := 0; m < middle; m++ {
			base := (o*middle + m) * inner
			out[o][m] = append([]float64(nil), t.Data[base:base+inner]...)
		}
	}
	return out
}
