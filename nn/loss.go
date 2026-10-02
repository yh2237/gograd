package nn

import (
	"fmt"
	"github.com/yh2237/gograd/tensor"
	"math"
)

// MaskedWeightedMSE computes mean weighted squared error over [B,T,C].
// frameWeight has B*T entries; zero excludes padding. channelWeight may be
// nil or have C entries. The denominator is the sum of element weights.
func MaskedWeightedMSE(pred, target *tensor.Tensor, frameWeight, channelWeight []float32) (float64, *tensor.Tensor, error) {
	return maskedLoss(pred, target, frameWeight, channelWeight, false)
}

// MaskedL1 computes mean weighted absolute error with the same mask semantics.
func MaskedL1(pred, target *tensor.Tensor, frameWeight, channelWeight []float32) (float64, *tensor.Tensor, error) {
	return maskedLoss(pred, target, frameWeight, channelWeight, true)
}
func maskedLoss(pred, target *tensor.Tensor, fw, cw []float32, l1 bool) (float64, *tensor.Tensor, error) {
	s := pred.Shape()
	ts := target.Shape()
	if len(s) != 3 || len(ts) != 3 || ts[0] != s[0] || ts[1] != s[1] || ts[2] != s[2] || target.Device() != pred.Device() || len(fw) != s[0]*s[1] || (cw != nil && len(cw) != s[2]) {
		return 0, nil, fmt.Errorf("nn: masked loss shape mismatch")
	}
	p, err := values(pred)
	if err != nil {
		return 0, nil, err
	}
	t, err := values(target)
	if err != nil {
		return 0, nil, err
	}
	g := make([]float32, len(p))
	var total, denom float64
	for i := range p {
		w := float64(fw[i/s[2]])
		if cw != nil {
			w *= float64(cw[i%s[2]])
		}
		if w < 0 {
			return 0, nil, fmt.Errorf("nn: negative loss weight")
		}
		d := float64(p[i] - t[i])
		if l1 {
			total += w * math.Abs(d)
			if d > 0 {
				g[i] = float32(w)
			} else if d < 0 {
				g[i] = -float32(w)
			}
		} else {
			total += w * d * d
			g[i] = float32(2 * w * d)
		}
		denom += w
	}
	if denom > 0 {
		for i := range g {
			g[i] /= float32(denom)
		}
	} else {
		denom = 1
	}
	gt, err := output(pred.Device(), s, g)
	return total / denom, gt, err
}
