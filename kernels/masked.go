package kernels

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

const maskedSource = `
extern "C" __global__ void masked_loss(const float* pred, const float* target, const float* frame, const float* channel, float* grad, float* loss, int n, int channels, int l1, float invDenom, int hasChannel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float w = frame[i / channels] * (hasChannel ? channel[i % channels] : 1.0f);
	float d = pred[i] - target[i];
	float v = l1 ? fabsf(d) : d * d;
	grad[i] = (l1 ? (d > 0.0f ? 1.0f : d < 0.0f ? -1.0f : 0.0f) : 2.0f * d) * w * invDenom;
	if (w != 0.0f) atomicAdd(loss, w * v * invDenom);
}
`

var maskedOnce sync.Once
var maskedKernel *cuda.Kernel
var maskedErr error

// MaskedLoss writes a weighted loss scalar and the gradient on the device.
func MaskedLoss(pred, target, frame, channel, grad, loss *cuda.Buffer, count, channels int, l1 bool, invDenom float32) error {
	maskedOnce.Do(func() {
		p, err := cuda.Compile(maskedSource)
		if err != nil {
			maskedErr = err
			return
		}
		maskedKernel, maskedErr = p.Function("masked_loss")
	})
	if maskedErr != nil {
		return maskedErr
	}
	p, t, f, g, l := pred.Pointer(), target.Pointer(), frame.Pointer(), grad.Pointer(), loss.Pointer()
	var c uintptr
	if channel != nil {
		c = channel.Pointer()
	}
	n, ch, l1v, has := int32(count), int32(channels), int32(0), int32(0)
	if l1 {
		l1v = 1
	}
	if channel != nil {
		has = 1
	}
	args := []unsafe.Pointer{unsafe.Pointer(&p), unsafe.Pointer(&t), unsafe.Pointer(&f), unsafe.Pointer(&c), unsafe.Pointer(&g), unsafe.Pointer(&l), unsafe.Pointer(&n), unsafe.Pointer(&ch), unsafe.Pointer(&l1v), unsafe.Pointer(&invDenom), unsafe.Pointer(&has)}
	return maskedKernel.Launch(elementGrid(count), [3]int{256, 1, 1}, 0, nil, args)
}
