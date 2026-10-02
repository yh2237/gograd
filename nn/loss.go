package nn

import (
	"fmt"
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
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
	if pred.Device() == tensor.CUDA {
		loss, err := NewMaskedLoss(tensor.CUDA, fw, cw, s[2], l1)
		if err != nil {
			return 0, nil, err
		}
		defer loss.Close()
		return loss.Forward(pred, target)
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

// MaskedLossOp keeps fixed loss weights on the device across training steps.
type MaskedLossOp struct {
	frame, channel, scalar *tensor.Tensor
	channels               int
	l1                     bool
	invDenom               float32
}

func NewMaskedLoss(device tensor.Device, fw, cw []float32, channels int, l1 bool) (*MaskedLossOp, error) {
	var denom float64
	for _, f := range fw {
		if f < 0 {
			return nil, fmt.Errorf("nn: negative loss weight")
		}
		if cw == nil {
			denom += float64(f) * float64(channels)
		} else {
			for _, c := range cw {
				if c < 0 {
					return nil, fmt.Errorf("nn: negative loss weight")
				}
				denom += float64(f) * float64(c)
			}
		}
	}
	if denom == 0 {
		denom = 1
	}
	f, err := tensor.FromHostOn(device, []int{len(fw)}, fw)
	if err != nil {
		return nil, err
	}
	var c *tensor.Tensor
	if cw != nil {
		c, err = tensor.FromHostOn(device, []int{len(cw)}, cw)
		if err != nil {
			f.Close()
			return nil, err
		}
	}
	s, err := tensor.ZerosOn(device, 1)
	if err != nil {
		f.Close()
		if c != nil {
			c.Close()
		}
		return nil, err
	}
	return &MaskedLossOp{frame: f, channel: c, scalar: s, channels: channels, l1: l1, invDenom: float32(1 / denom)}, nil
}

func (op *MaskedLossOp) Close() {
	op.frame.Close()
	if op.channel != nil {
		op.channel.Close()
	}
	op.scalar.Close()
}

func (op *MaskedLossOp) Forward(pred, target *tensor.Tensor) (float64, *tensor.Tensor, error) {
	if pred.Device() == tensor.CPU {
		return maskedLoss(pred, target, op.frame.Data(), func() []float32 {
			if op.channel != nil {
				return op.channel.Data()
			}
			return nil
		}(), op.l1)
	}
	s := pred.Shape()
	if len(s) != 3 || s[2] != op.channels || s[0]*s[1] != op.frame.Numel() || target.Numel() != pred.Numel() || target.Device() != pred.Device() {
		return 0, nil, fmt.Errorf("nn: masked loss shape mismatch")
	}
	g, err := tensor.NewOn(tensor.CUDA, s...)
	if err != nil {
		return 0, nil, err
	}
	var channel *cuda.Buffer
	if op.channel != nil {
		channel = op.channel.Buffer()
	}
	if err = op.scalar.Zero(); err == nil {
		err = kernels.MaskedLoss(pred.Buffer(), target.Buffer(), op.frame.Buffer(), channel, g.Buffer(), op.scalar.Buffer(), pred.Numel(), op.channels, op.l1, op.invDenom)
	}
	if err != nil {
		g.Close()
		return 0, nil, err
	}
	v, err := op.scalar.ToHost()
	if err != nil {
		g.Close()
		return 0, nil, err
	}
	return float64(v[0]), g, nil
}
