package autograd

import (
	"fmt"
	"math"
	"slices"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// SGDOptions follows PyTorch SGD: coupled weight decay, momentum/dampening,
// optional Nesterov and gradient ascent. First momentum buffer equals the full
// decay-adjusted gradient, without dampening. Nesterov needs positive momentum
// and zero dampening. All numeric options must be finite/nonnegative.
type SGDOptions struct {
	LR, Momentum, Dampening, WeightDecay float32
	Nesterov, Maximize                   bool
}

func (o SGDOptions) validate() error {
	for _, v := range []float32{o.LR, o.Momentum, o.Dampening, o.WeightDecay} {
		if v < 0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("autograd: invalid SGD configuration")
		}
	}
	if o.Nesterov && (o.Momentum <= 0 || o.Dampening != 0) {
		return fmt.Errorf("autograd: Nesterov needs positive momentum and zero dampening")
	}
	return nil
}

type SGD struct {
	Params                    []Parameter
	LR                        float32
	StepCount                 int
	options                   SGDOptions
	buffers                   [][]float32
	initialized               []bool
	bufferGPU, initializedGPU []*cuda.Buffer
	graphStep                 *cuda.Buffer
	closed                    bool
}

func validateSGDParameters(params []Parameter) error {
	seenNames := make(map[string]bool)
	seenStorage := make(map[*tensorStorage]bool)
	var device tensor.Device
	for i, p := range params {
		if p.Name == "" || seenNames[p.Name] || p.Value == nil {
			return fmt.Errorf("autograd: SGD needs distinct named parameters")
		}
		if err := p.Value.checkOpen(); err != nil {
			return err
		}
		v := p.Value
		if v.intermediate || v.ephemeral || !v.IsContiguous() || v.DType != Float32 || v.Numel() < 1 || v.Numel() > math.MaxInt32 || seenStorage[v.storage] {
			return fmt.Errorf("autograd: SGD requires distinct contiguous float32 leaves")
		}
		if i == 0 {
			device = v.Device
		}
		if v.Device != device {
			return fmt.Errorf("autograd: SGD parameters must use one device")
		}
		seenNames[p.Name], seenStorage[v.storage] = true, true
	}
	return nil
}

func NewSGD(params []Parameter, options SGDOptions) (*SGD, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}
	if err := validateSGDParameters(params); err != nil {
		return nil, err
	}
	o := &SGD{Params: slices.Clone(params), LR: options.LR, options: options, buffers: make([][]float32, len(params)), initialized: make([]bool, len(params))}
	if len(params) > 0 && params[0].Value.Device == tensor.CUDA && options.Momentum > 0 {
		var err error
		o.bufferGPU, o.initializedGPU, err = sgdDeviceStorage(params)
		if err != nil {
			o.Close()
			return nil, err
		}
	}
	return o, nil
}
func (o *SGD) LearningRate() float32 { return o.LR }
func (o *SGD) SetLearningRate(lr float32) {
	if err := validateLearningRate(lr); err != nil {
		panic(err)
	}
	o.LR = lr
}
func (o *SGD) config() SGDOptions { config := o.options; config.LR = o.LR; return config }
func (o *SGD) validateOpen() error {
	if o == nil || o.closed {
		return fmt.Errorf("autograd: SGD is closed")
	}
	if err := o.config().validate(); err != nil {
		return err
	}
	if len(o.buffers) != len(o.Params) || len(o.initialized) != len(o.Params) {
		return fmt.Errorf("autograd: SGD parameter storage count changed")
	}
	if len(o.Params) > 0 && o.Params[0].Value != nil && o.Params[0].Value.Device == tensor.CUDA && o.options.Momentum > 0 && (len(o.bufferGPU) != len(o.Params) || len(o.initializedGPU) != len(o.Params)) {
		return fmt.Errorf("autograd: incomplete SGD device storage")
	}
	return validateSGDParameters(o.Params)
}
func (o *SGD) Step() {
	if err := o.validateOpen(); err != nil {
		panic(err)
	}
	for _, p := range o.Params {
		v := p.Value
		if v.Device == tensor.CPU && v.Grad != nil && len(v.Grad) != v.Numel() || v.Device == tensor.CUDA && v.gradBuf != nil && v.gradBuf.Size() != v.Numel()*4 {
			panic("autograd: SGD gradient shape mismatch")
		}
	}
	if cuda.InCapture() {
		if o.graphStep == nil {
			panic("autograd: call SGD.PrepareGraph before capture")
		}
		o.stepGPU(true)
		return
	}
	if o.graphStep != nil {
		if err := o.SyncGraphStepCount(); err != nil {
			panic(err)
		}
	}
	if o.StepCount >= math.MaxInt32 {
		panic("autograd: SGD step count overflow")
	}
	o.StepCount++
	if len(o.Params) > 0 && o.Params[0].Value.Device == tensor.CUDA {
		o.stepGPU(false)
		if o.graphStep != nil {
			if err := o.PrepareGraph(); err != nil {
				panic(err)
			}
		}
		return
	}
	for index, p := range o.Params {
		v := p.Value
		if v.Grad == nil {
			continue
		}
		first := !o.initialized[index]
		if o.options.Momentum > 0 && first {
			o.buffers[index] = make([]float32, v.Numel())
		}
		for i, g := range v.Grad {
			if o.options.Maximize {
				g = -g
			}
			g += float32(o.options.WeightDecay * v.Data[i])
			if o.options.Momentum > 0 {
				if first {
					o.buffers[index][i] = g
				} else {
					o.buffers[index][i] = float32(o.options.Momentum*o.buffers[index][i]) + float32((1-o.options.Dampening)*g)
				}
				if o.options.Nesterov {
					g += float32(o.options.Momentum * o.buffers[index][i])
				} else {
					g = o.buffers[index][i]
				}
			}
			v.Data[i] -= float32(o.LR * g)
		}
		if o.options.Momentum > 0 {
			o.initialized[index] = true
		}
		v.storage.version.Add(1)
	}
}

// Eager SGD resets gradients to absent on both devices, matching PyTorch's
// set_to_none behavior. Prepared/captured graphs preserve CUDA gradient buffers
// and zero them, since replay needs stable addresses.
func (o *SGD) ZeroGrad() {
	if err := o.validateOpen(); err != nil {
		panic(err)
	}
	for _, p := range o.Params {
		if o.graphStep != nil || cuda.InCapture() {
			p.Value.ZeroGrad()
			continue
		}
		p.Value.Grad = nil
		if p.Value.gradBuf != nil {
			p.Value.gradBuf.Free()
			p.Value.gradBuf = nil
		}
	}
}
func (o *SGD) Close() {
	if o.closed {
		return
	}
	sgdFreeStorage(o.bufferGPU, o.initializedGPU)
	if o.graphStep != nil {
		o.graphStep.Free()
	}
	o.buffers, o.initialized, o.bufferGPU, o.initializedGPU, o.graphStep = nil, nil, nil, nil, nil
	o.closed = true
}
