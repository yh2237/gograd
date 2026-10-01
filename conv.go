package gograd

// Conv1d computes a dilated, same-length convolution with symmetric zero
// padding, matching torch.nn.Conv1d with padding=dilation and kernel size K.
// x is [B,C,T], weight is [O,C,K], bias is [O], and the result is [B,O,T].
func Conv1d(x, weight, bias *Tensor, dilation int) *Tensor {
	if x.Dims() != 3 || weight.Dims() != 3 || bias.Dims() != 1 {
		panic("gograd: Conv1d expects x[B,C,T], weight[O,C,K], bias[O]")
	}
	batch, channels, time := x.Shape[0], x.Shape[1], x.Shape[2]
	outChannels := weight.Shape[0]
	kernel := weight.Shape[2]
	if weight.Shape[1] != channels || bias.Shape[0] != outChannels {
		panic("gograd: Conv1d shape mismatch")
	}
	if dilation < 1 {
		panic("gograd: Conv1d dilation must be positive")
	}
	out := NewTensor([]int{batch, outChannels, time}, make([]float64, batch*outChannels*time))
	for b := 0; b < batch; b++ {
		for o := 0; o < outChannels; o++ {
			for t := 0; t < time; t++ {
				sum := bias.Data[o]
				for c := 0; c < channels; c++ {
					for k := 0; k < kernel; k++ {
						index := t - dilation + k*dilation
						if index < 0 || index >= time {
							continue
						}
						sum += weight.Data[(o*channels+c)*kernel+k] *
							x.Data[(b*channels+c)*time+index]
					}
				}
				out.Data[(b*outChannels+o)*time+t] = sum
			}
		}
	}
	out.parents = []*Tensor{x, weight, bias}
	out.backward = func() {
		for b := 0; b < batch; b++ {
			for o := 0; o < outChannels; o++ {
				for t := 0; t < time; t++ {
					g := out.Grad[(b*outChannels+o)*time+t]
					if g == 0 {
						continue
					}
					bias.Grad[o] += g
					for c := 0; c < channels; c++ {
						for k := 0; k < kernel; k++ {
							index := t - dilation + k*dilation
							if index < 0 || index >= time {
								continue
							}
							wi := (o*channels+c)*kernel + k
							weight.Grad[wi] += g * x.Data[(b*channels+c)*time+index]
							x.Grad[(b*channels+c)*time+index] += g * weight.Data[wi]
						}
					}
				}
			}
		}
	}
	return out
}
