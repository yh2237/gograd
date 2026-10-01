// Package gputcn runs the frame intonation TCN forward and backward passes on
// the GPU with the cuda package. Tensors are row-major float32 device buffers
// and the layout follows the PyTorch model: transpose to channel-major for
// each convolution and back for the linear layers.
package gputcn

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

// Layer is one dilated residual convolution.
type Layer struct {
	Weight   *cuda.Buffer // [hidden,hidden,3]
	Bias     *cuda.Buffer // [hidden]
	Dilation int
}

// Model holds the device parameters of a FrameIntonationTCN.
type Model struct {
	Inputs       int
	Hidden       int
	Layers       []Layer
	InputWeight  *cuda.Buffer // [hidden,inputs]
	InputBias    *cuda.Buffer // [hidden]
	OutputWeight *cuda.Buffer // [1,hidden]
	OutputBias   *cuda.Buffer // [1]
}

// Close releases every device parameter.
func (m *Model) Close() {
	for _, layer := range m.Layers {
		layer.Weight.Free()
		layer.Bias.Free()
	}
	m.InputWeight.Free()
	m.InputBias.Free()
	m.OutputWeight.Free()
	m.OutputBias.Free()
}

// UploadFloat32 copies a host slice into a new device buffer.
func UploadFloat32(values []float32) (*cuda.Buffer, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("gputcn: empty upload")
	}
	buffer, err := cuda.Alloc(len(values) * 4)
	if err != nil {
		return nil, err
	}
	if err := buffer.CopyFromHost(floatsToBytes(values)); err != nil {
		buffer.Free()
		return nil, err
	}
	return buffer, nil
}

// DownloadFloat32 copies a device buffer into a host slice.
func DownloadFloat32(buffer *cuda.Buffer, count int) ([]float32, error) {
	result := make([]float32, count)
	if err := buffer.CopyToHost(floatsToBytes(result)); err != nil {
		return nil, err
	}
	return result, nil
}

type layerCache struct {
	inputChannelMajor *cuda.Buffer // transpose of the layer input [batch,hidden,time]
	outputActivation  *cuda.Buffer // activation after the residual tanh [batch,time,hidden]
}

// Cache holds the activations saved by a forward pass for backpropagation.
type Cache struct {
	rows            int
	batch           int
	time            int
	input           *cuda.Buffer
	inputActivation *cuda.Buffer
	layers          []layerCache
	output          *cuda.Buffer
	buffers         []*cuda.Buffer
}

// Output returns the [batch,time] device output buffer.
func (c *Cache) Output() *cuda.Buffer { return c.output }

// Close releases the saved activations.
func (c *Cache) Close() {
	for _, buffer := range c.buffers {
		buffer.Free()
	}
	c.buffers = nil
}

func (c *Cache) alloc(count int) (*cuda.Buffer, error) {
	buffer, err := cuda.Alloc(count * 4)
	if err != nil {
		return nil, err
	}
	c.buffers = append(c.buffers, buffer)
	return buffer, nil
}

func (c *Cache) zeroAlloc(count int) (*cuda.Buffer, error) {
	buffer, err := c.alloc(count)
	if err != nil {
		return nil, err
	}
	if err := buffer.Memset(0, buffer.Size()); err != nil {
		return nil, err
	}
	return buffer, nil
}

// ForwardCached runs the model and keeps the activations needed for Backward.
func (m *Model) ForwardCached(x []float32, batch, time int) (*Cache, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		return nil, err
	}
	blas, err := cuda.NewBlas()
	if err != nil {
		return nil, err
	}
	defer blas.Destroy()

	cache := &Cache{rows: batch * time, batch: batch, time: time}
	fail := func(err error) (*Cache, error) {
		cache.Close()
		return nil, err
	}
	rows := cache.rows
	hidden := m.Hidden

	input, err := cache.alloc(rows * m.Inputs)
	if err != nil {
		return fail(err)
	}
	cache.input = input
	if err := input.CopyFromHost(floatsToBytes(x)); err != nil {
		return fail(err)
	}

	state, err := cache.alloc(rows * hidden)
	if err != nil {
		return fail(err)
	}
	if err := blas.SgemmRowMajorNT(rows, hidden, m.Inputs, 1, input.Pointer(), m.Inputs, m.InputWeight.Pointer(), m.Inputs, 0, state.Pointer(), hidden); err != nil {
		return fail(err)
	}
	if err := cuda.BiasColumnsTanh(state, m.InputBias, rows, hidden); err != nil {
		return fail(err)
	}
	cache.inputActivation = state

	for index := range m.Layers {
		layer := &m.Layers[index]
		channelMajor, err := cache.alloc(batch * hidden * time)
		if err != nil {
			return fail(err)
		}
		if err := cuda.Transpose12(state, channelMajor, batch, time, hidden); err != nil {
			return fail(err)
		}
		convolved, err := cache.alloc(batch * hidden * time)
		if err != nil {
			return fail(err)
		}
		if err := cuda.Conv1dForward(blas, channelMajor, layer.Weight, layer.Bias, convolved, batch, hidden, time, hidden, 3, layer.Dilation); err != nil {
			return fail(err)
		}
		next, err := cache.alloc(batch * time * hidden)
		if err != nil {
			return fail(err)
		}
		if err := cuda.TransposeAddTanh(convolved, state, next, batch, hidden, time); err != nil {
			return fail(err)
		}
		cache.layers = append(cache.layers, layerCache{inputChannelMajor: channelMajor, outputActivation: next})
		state = next
	}

	output, err := cache.alloc(rows)
	if err != nil {
		return fail(err)
	}
	if err := blas.SgemmRowMajorNT(rows, 1, hidden, 1, state.Pointer(), hidden, m.OutputWeight.Pointer(), hidden, 0, output.Pointer(), 1); err != nil {
		return fail(err)
	}
	if err := cuda.AddBiasColumns(output, m.OutputBias, rows, 1); err != nil {
		return fail(err)
	}
	if err := cuda.Synchronize(); err != nil {
		return fail(err)
	}
	cache.output = output
	return cache, nil
}

// Forward is a convenience wrapper that returns the output and frees the cache.
func (m *Model) Forward(x []float32, batch, time int) ([]float32, error) {
	cache, err := m.ForwardCached(x, batch, time)
	if err != nil {
		return nil, err
	}
	defer cache.Close()
	if err := cuda.Synchronize(); err != nil {
		return nil, err
	}
	return DownloadFloat32(cache.output, batch*time)
}

// GradientLayer holds device gradients for one layer.
type GradientLayer struct {
	Weight *cuda.Buffer
	Bias   *cuda.Buffer
}

// Gradients holds device parameter gradients.
type Gradients struct {
	InputWeight  *cuda.Buffer
	InputBias    *cuda.Buffer
	Layers       []GradientLayer
	OutputWeight *cuda.Buffer
	OutputBias   *cuda.Buffer
	buffers      []*cuda.Buffer
}

// Close releases the gradient buffers.
func (g *Gradients) Close() {
	for _, buffer := range g.buffers {
		buffer.Free()
	}
	g.buffers = nil
}

func (g *Gradients) alloc(count int) (*cuda.Buffer, error) {
	buffer, err := cuda.Alloc(count * 4)
	if err != nil {
		return nil, err
	}
	g.buffers = append(g.buffers, buffer)
	return buffer, nil
}

func (g *Gradients) zeroAlloc(count int) (*cuda.Buffer, error) {
	buffer, err := g.alloc(count)
	if err != nil {
		return nil, err
	}
	if err := buffer.Memset(0, buffer.Size()); err != nil {
		buffer.Free()
		return nil, err
	}
	return buffer, nil
}

// Backward propagates dy [batch,time] through the cached forward pass and
// returns the parameter gradients.
func (m *Model) Backward(cache *Cache, dy *cuda.Buffer) (*Gradients, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.SetDevice(0); err != nil {
		return nil, err
	}
	blas, err := cuda.NewBlas()
	if err != nil {
		return nil, err
	}
	defer blas.Destroy()

	rows := cache.rows
	batch, time := cache.batch, cache.time
	hidden := m.Hidden

	grads := &Gradients{}
	fail := func(err error) (*Gradients, error) {
		grads.Close()
		return nil, err
	}

	lastActivation := cache.inputActivation
	if len(cache.layers) > 0 {
		lastActivation = cache.layers[len(cache.layers)-1].outputActivation
	}
	outputWeight, err := grads.alloc(hidden)
	if err != nil {
		return fail(err)
	}
	grads.OutputWeight = outputWeight
	outputBias, err := grads.zeroAlloc(1)
	if err != nil {
		return fail(err)
	}
	grads.OutputBias = outputBias
	if err := blas.SgemmRowMajorTransposeA(1, hidden, rows, 1, dy.Pointer(), 1, lastActivation.Pointer(), hidden, 0, outputWeight.Pointer(), hidden); err != nil {
		return fail(err)
	}
	if err := cuda.ColumnSum(dy, outputBias, rows, 1); err != nil {
		return fail(err)
	}

	dA, err := grads.alloc(rows * hidden)
	if err != nil {
		return fail(err)
	}
	if err := blas.SgemmRowMajor(rows, hidden, 1, 1, dy.Pointer(), 1, m.OutputWeight.Pointer(), hidden, 0, dA.Pointer(), hidden); err != nil {
		return fail(err)
	}

	grads.Layers = make([]GradientLayer, len(m.Layers))
	for index := len(m.Layers) - 1; index >= 0; index-- {
		layer := &m.Layers[index]
		cacheLayer := cache.layers[index]
		dZ, err := grads.zeroAlloc(rows * hidden)
		if err != nil {
			return fail(err)
		}
		if err := cuda.TanhBackward(cacheLayer.outputActivation, dA, dZ, rows*hidden); err != nil {
			return fail(err)
		}

		// dZ is already the gradient of the transposed convolution output, so
		// transpose it directly instead of copying into another buffer.
		dConvChannel, err := grads.alloc(batch * hidden * time)
		if err != nil {
			return fail(err)
		}
		if err := cuda.Transpose12(dZ, dConvChannel, batch, time, hidden); err != nil {
			return fail(err)
		}
		weightGrad, err := grads.zeroAlloc(hidden * hidden * 3)
		if err != nil {
			return fail(err)
		}
		biasGrad, err := grads.zeroAlloc(hidden)
		if err != nil {
			return fail(err)
		}
		grads.Layers[index] = GradientLayer{Weight: weightGrad, Bias: biasGrad}
		if err := cuda.ConvWeightGrad(cacheLayer.inputChannelMajor, dConvChannel, weightGrad, batch, hidden, time, hidden, 3, layer.Dilation); err != nil {
			return fail(err)
		}
		if err := cuda.ConvBiasGrad(dConvChannel, biasGrad, batch, hidden, time); err != nil {
			return fail(err)
		}
		dInputChannel, err := grads.zeroAlloc(batch * hidden * time)
		if err != nil {
			return fail(err)
		}
		if err := cuda.ConvInputGrad(dConvChannel, layer.Weight, dInputChannel, batch, hidden, time, hidden, 3, layer.Dilation); err != nil {
			return fail(err)
		}
		dInputTime, err := grads.alloc(batch * time * hidden)
		if err != nil {
			return fail(err)
		}
		if err := cuda.Transpose12(dInputChannel, dInputTime, batch, hidden, time); err != nil {
			return fail(err)
		}
		dAPrev, err := grads.alloc(rows * hidden)
		if err != nil {
			return fail(err)
		}
		if err := cuda.AddPair(dAPrev, dZ, dInputTime, rows*hidden); err != nil {
			return fail(err)
		}
		dA = dAPrev
	}

	dZ1, err := grads.zeroAlloc(rows * hidden)
	if err != nil {
		return fail(err)
	}
	if err := cuda.TanhBackward(cache.inputActivation, dA, dZ1, rows*hidden); err != nil {
		return fail(err)
	}
	inputWeight, err := grads.alloc(hidden * m.Inputs)
	if err != nil {
		return fail(err)
	}
	grads.InputWeight = inputWeight
	inputBias, err := grads.zeroAlloc(hidden)
	if err != nil {
		return fail(err)
	}
	grads.InputBias = inputBias
	if err := blas.SgemmRowMajorTransposeA(hidden, m.Inputs, rows, 1, dZ1.Pointer(), hidden, cache.input.Pointer(), m.Inputs, 0, inputWeight.Pointer(), m.Inputs); err != nil {
		return fail(err)
	}
	if err := cuda.ColumnSum(dZ1, inputBias, rows, hidden); err != nil {
		return fail(err)
	}
	if err := cuda.Synchronize(); err != nil {
		return fail(err)
	}
	return grads, nil
}

func floatsToBytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
