package autograd

// RMSNormLast normalizes the final axis and applies a learned channel weight.
// It is a graph composition, so forward and backward use the selected backend.
func RMSNormLast(x, weight *Tensor, eps float32) *Tensor {
	dispatchBackend("rms_norm_last", x.Device)
	if len(x.Shape) == 0 || len(weight.Shape) != 1 || weight.Shape[0] != x.Shape[len(x.Shape)-1] || eps <= 0 || x.Device != weight.Device {
		panic("autograd: invalid RMSNorm shape, device, or epsilon")
	}
	meanSquare := Mean(Mul(x, x), len(x.Shape)-1)
	inv := Exp(MulScalar(Log(AddScalar(meanSquare, eps)), -0.5))
	shape := append(append([]int(nil), meanSquare.Shape...), 1)
	return Mul(Mul(x, Reshape(inv, shape...)), weight)
}
