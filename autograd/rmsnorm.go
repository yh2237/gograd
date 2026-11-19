package autograd

// RMSNormLast normalizes the final axis and applies a learned channel weight.
// It is a graph composition, so forward and backward use the selected backend.
func RMSNormLast(x, weight *Tensor, eps float32) *Tensor {
	if inferenceDevice(x.Device) {
		inferenceOperands(x, weight)
	} else {
		dispatchBackend("rms_norm_last", x.Device)
		same(x, weight)
	}
	if len(x.Shape) == 0 || len(weight.Shape) != 1 || weight.Shape[0] != x.Shape[len(x.Shape)-1] || eps <= 0 || x.Device != weight.Device {
		panic("autograd: invalid RMSNorm shape, device, or epsilon")
	}
	var owned []*Tensor
	keep := func(t *Tensor) *Tensor { owned = append(owned, t); return t }
	defer func() {
		if r := recover(); r != nil {
			for i := len(owned) - 1; i >= 0; i-- {
				owned[i].Close()
			}
			panic(r)
		}
	}()
	sq := keep(Mul(x, x))
	meanSquare := keep(Mean(sq, len(x.Shape)-1))
	shifted := keep(AddScalar(meanSquare, eps))
	logged := keep(Log(shifted))
	scaled := keep(MulScalar(logged, -0.5))
	inv := keep(Exp(scaled))
	shape := append(append([]int(nil), meanSquare.Shape...), 1)
	invView := keep(Reshape(inv, shape...))
	normed := keep(Mul(x, invView))
	return Mul(normed, weight)
}
