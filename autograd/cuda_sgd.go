package autograd

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

const sgdSource = `
extern "C" __global__ void sgd_update(float* p,const float* grad,float* buffer,const float* initialized,int n,float lr,float momentum,float dampening,float decay,int nesterov,int maximize){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float g=maximize?-grad[i]:grad[i];g+=decay*p[i];
 if(momentum>0){float b=initialized[0]==0?g:momentum*buffer[i]+(1-dampening)*g;buffer[i]=b;g=nesterov?g+momentum*b:b;}
 p[i]-=lr*g;
}
extern "C" __global__ void sgd_mark(float* initialized){if(threadIdx.x==0&&blockIdx.x==0)initialized[0]=1;}
extern "C" __global__ void sgd_step(int* step){if(threadIdx.x==0&&blockIdx.x==0)step[0]+=1;}
`

var sgdOnce sync.Once
var sgdProgram *cuda.Program
var sgdKernels map[string]*cuda.Kernel
var sgdCompileErr error

func launchSGD(name string, n int, args ...unsafe.Pointer) {
	sgdOnce.Do(func() {
		sgdProgram, sgdCompileErr = cuda.Compile(sgdSource)
		if sgdCompileErr != nil {
			return
		}
		sgdKernels = make(map[string]*cuda.Kernel)
		for _, key := range []string{"sgd_update", "sgd_mark", "sgd_step"} {
			sgdKernels[key], sgdCompileErr = sgdProgram.Function(key)
			if sgdCompileErr != nil {
				return
			}
		}
	})
	if sgdCompileErr != nil {
		panic(sgdCompileErr)
	}
	timedGPU(name, func() error {
		return sgdKernels[name].Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, nil, args)
	})
}
func sgdFreeStorage(buffers, flags []*cuda.Buffer) {
	for _, b := range append(buffers, flags...) {
		if b != nil {
			b.Free()
		}
	}
}
func sgdDeviceStorage(params []Parameter) (buffers, flags []*cuda.Buffer, err error) {
	defer func() {
		if err != nil {
			sgdFreeStorage(buffers, flags)
			buffers, flags = nil, nil
		}
	}()
	for _, p := range params {
		b, e := cuda.Alloc(p.Value.Numel() * 4)
		if e != nil {
			return buffers, flags, e
		}
		buffers = append(buffers, b)
		if e := b.Memset(0, b.Size()); e != nil {
			return buffers, flags, e
		}
		f, e := cuda.Alloc(4)
		if e != nil {
			return buffers, flags, e
		}
		flags = append(flags, f)
		if e := f.Memset(0, 4); e != nil {
			return buffers, flags, e
		}
	}
	return buffers, flags, nil
}
func (o *SGD) stepGPU(captured bool) {
	if captured {
		step := ptr(o.graphStep)
		launchSGD("sgd_step", 1, unsafe.Pointer(&step))
	}
	for i, p := range o.Params {
		v := p.Value
		if v.gradBuf == nil {
			continue
		}
		pp, gp := ptr(v.buf), ptr(v.gradBuf)
		var bp, fp uintptr
		if o.options.Momentum > 0 {
			bp, fp = ptr(o.bufferGPU[i]), ptr(o.initializedGPU[i])
		}
		n, nesterov, maximize := int32(v.Numel()), int32(0), int32(0)
		if o.options.Nesterov {
			nesterov = 1
		}
		if o.options.Maximize {
			maximize = 1
		}
		launchSGD("sgd_update", v.Numel(), unsafe.Pointer(&pp), unsafe.Pointer(&gp), unsafe.Pointer(&bp), unsafe.Pointer(&fp), unsafe.Pointer(&n),
			unsafe.Pointer(&o.LR), unsafe.Pointer(&o.options.Momentum), unsafe.Pointer(&o.options.Dampening), unsafe.Pointer(&o.options.WeightDecay), unsafe.Pointer(&nesterov), unsafe.Pointer(&maximize))
		if o.options.Momentum > 0 {
			launchSGD("sgd_mark", 1, unsafe.Pointer(&fp))
		}
		if !captured {
			v.invalidateBF16()
		}
	}
}
func (o *SGD) PrepareGraph() error {
	if err := o.validateOpen(); err != nil {
		return err
	}
	if len(o.Params) == 0 || o.Params[0].Value.Device != tensor.CUDA {
		return cuda.ErrUnavailable
	}
	if o.graphStep == nil {
		b, err := cuda.Alloc(4)
		if err != nil {
			return err
		}
		o.graphStep = b
	}
	step := int32(o.StepCount)
	return o.graphStep.CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&step)), 4))
}
func (o *SGD) GraphStepCount() (int, error) {
	if o.closed || o.graphStep == nil {
		return 0, cuda.ErrUnavailable
	}
	var step int32
	if err := o.graphStep.CopyToHost(unsafe.Slice((*byte)(unsafe.Pointer(&step)), 4)); err != nil {
		return 0, err
	}
	return int(step), nil
}
func (o *SGD) SyncGraphStepCount() error {
	step, err := o.GraphStepCount()
	if err != nil {
		return err
	}
	o.StepCount = step
	return nil
}
