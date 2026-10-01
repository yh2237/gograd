// Package gputcn runs the frame intonation TCN forward and backward passes on
// the GPU with the cuda package. Tensors are row-major float32 device buffers
// and the layout follows the PyTorch model: transpose to channel-major for
// each convolution and back for the linear layers.
package gputcn

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
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

	blas *cuda.Blas
}

// NewFrameIntonationTCN builds a model with uniformly initialized parameters
// and returns it. The RNG is local, so runs are reproducible.
func NewFrameIntonationTCN(inputs, hidden int, dilations []int, seed int64) (*Model, error) {
	rng := rand.New(rand.NewSource(seed))
	uniform := func(count, fanIn int) (*cuda.Buffer, error) {
		values := make([]float32, count)
		bound := float32(1 / math.Sqrt(float64(fanIn)))
		for i := range values {
			values[i] = (rng.Float32()*2 - 1) * bound
		}
		return UploadFloat32(values)
	}
	model := &Model{Inputs: inputs, Hidden: hidden}
	var err error
	if model.InputWeight, err = uniform(hidden*inputs, inputs); err != nil {
		return nil, err
	}
	if model.InputBias, err = uniform(hidden, inputs); err != nil {
		model.Close()
		return nil, err
	}
	for _, dilation := range dilations {
		weight, err := uniform(hidden*hidden*3, hidden*3)
		if err != nil {
			model.Close()
			return nil, err
		}
		bias, err := uniform(hidden, hidden*3)
		if err != nil {
			weight.Free()
			model.Close()
			return nil, err
		}
		model.Layers = append(model.Layers, Layer{Weight: weight, Bias: bias, Dilation: dilation})
	}
	if model.OutputWeight, err = uniform(hidden, hidden); err != nil {
		model.Close()
		return nil, err
	}
	if model.OutputBias, err = uniform(1, hidden); err != nil {
		model.Close()
		return nil, err
	}
	return model, nil
}

// blasHandle returns a cached cuBLAS handle. Creating a handle is comparatively
// expensive, so it is reused across forward and backward calls.
func (m *Model) blasHandle() (*cuda.Blas, error) {
	if m.blas != nil {
		return m.blas, nil
	}
	blas, err := cuda.NewBlas()
	if err != nil {
		return nil, err
	}
	m.blas = blas
	return blas, nil
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
	if m.blas != nil {
		m.blas.Destroy()
		m.blas = nil
	}
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
	outputActivation *cuda.Buffer // activation after the residual tanh [batch,time,hidden]
	columns          *cuda.Buffer // im2col of the layer input [hidden*3, batch*time]
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

// ensureDevice makes the CUDA context current unless a capture is in progress,
// which forbids setting the device.
func (m *Model) ensureDevice() error {
	if cuda.InCapture() {
		return nil
	}
	return cuda.SetDevice(0)
}

// SetStream binds the cached cuBLAS handle to a stream, as graph capture
// requires.
func (m *Model) SetStream(stream *cuda.Stream) error {
	blas, err := m.blasHandle()
	if err != nil {
		return err
	}
	return blas.SetStream(stream)
}

// ForwardCached runs the model on x and keeps the activations for Backward.
func (m *Model) ForwardCached(x []float32, batch, time int) (*Cache, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := m.ensureDevice(); err != nil {
		return nil, err
	}
	cache := &Cache{rows: batch * time, batch: batch, time: time}
	input, err := cache.alloc(cache.rows * m.Inputs)
	if err != nil {
		cache.Close()
		return nil, err
	}
	if err := input.CopyFromHost(floatsToBytes(x)); err != nil {
		cache.Close()
		return nil, err
	}
	if err := m.forwardFrom(cache, input); err != nil {
		cache.Close()
		return nil, err
	}
	return cache, nil
}

// ForwardCachedDevice is ForwardCached with a caller-owned input buffer, so a
// captured graph can read an input the host updates between launches.
func (m *Model) ForwardCachedDevice(input *cuda.Buffer, batch, time int) (*Cache, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := m.ensureDevice(); err != nil {
		return nil, err
	}
	cache := &Cache{rows: batch * time, batch: batch, time: time}
	if input.Size() < cache.rows*m.Inputs*4 {
		return nil, fmt.Errorf("gputcn: input buffer is too small")
	}
	if err := m.forwardFrom(cache, input); err != nil {
		cache.Close()
		return nil, err
	}
	return cache, nil
}

func (m *Model) forwardFrom(cache *Cache, input *cuda.Buffer) error {
	blas, err := m.blasHandle()
	if err != nil {
		return err
	}
	cache.input = input
	rows := cache.rows
	hidden := m.Hidden
	batch, time := cache.batch, cache.time

	state, err := cache.alloc(rows * hidden)
	if err != nil {
		return err
	}
	if err := blas.SgemmRowMajorNT(rows, hidden, m.Inputs, 1, input.Pointer(), m.Inputs, m.InputWeight.Pointer(), m.Inputs, 0, state.Pointer(), hidden); err != nil {
		return err
	}
	if err := kernels.BiasColumnsTanh(state, m.InputBias, rows, hidden); err != nil {
		return err
	}
	cache.inputActivation = state

	for index := range m.Layers {
		layer := &m.Layers[index]
		columns, err := cache.alloc(hidden * 3 * batch * time)
		if err != nil {
			return err
		}
		convolved, err := cache.alloc(hidden * batch * time)
		if err != nil {
			return err
		}
		if err := kernels.Conv1dForwardTo(blas, state, layer.Weight, convolved, columns, batch, hidden, time, hidden, 3, layer.Dilation); err != nil {
			return err
		}
		next, err := cache.alloc(batch * time * hidden)
		if err != nil {
			return err
		}
		if err := kernels.TransposeAddTanh(convolved, state, layer.Bias, next, batch, hidden, time); err != nil {
			return err
		}
		cache.layers = append(cache.layers, layerCache{outputActivation: next, columns: columns})
		state = next
	}

	output, err := cache.alloc(rows)
	if err != nil {
		return err
	}
	if err := blas.SgemmRowMajorNT(rows, 1, hidden, 1, state.Pointer(), hidden, m.OutputWeight.Pointer(), hidden, 0, output.Pointer(), 1); err != nil {
		return err
	}
	if err := kernels.AddBiasColumns(output, m.OutputBias, rows, 1); err != nil {
		return err
	}
	// Work stays queued on the stream; later kernels and the final copy are
	// ordered after it, so an explicit device sync here would only stall.
	cache.output = output
	return nil
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
	if err := m.ensureDevice(); err != nil {
		return nil, err
	}
	blas, err := m.blasHandle()
	if err != nil {
		return nil, err
	}

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
	if err := kernels.ColumnSum(dy, outputBias, rows, 1); err != nil {
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
		if err := kernels.TanhBackward(cacheLayer.outputActivation, dA, dZ, rows*hidden); err != nil {
			return fail(err)
		}

		dConv, err := grads.alloc(hidden * batch * time)
		if err != nil {
			return fail(err)
		}
		if err := kernels.ToHBT(dZ, dConv, batch, hidden, time); err != nil {
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
		if err := kernels.ConvWeightGradGemm(blas, dConv, cacheLayer.columns, weightGrad, batch, hidden, time, hidden, 3); err != nil {
			return fail(err)
		}
		if err := kernels.RowSum(dConv, biasGrad, hidden, batch*time); err != nil {
			return fail(err)
		}
		dInputTime, err := grads.zeroAlloc(rows * hidden)
		if err != nil {
			return fail(err)
		}
		if err := kernels.ConvInputGrad(blas, dConv, layer.Weight, dInputTime, batch, hidden, time, hidden, 3, layer.Dilation); err != nil {
			return fail(err)
		}
		dAPrev, err := grads.alloc(rows * hidden)
		if err != nil {
			return fail(err)
		}
		if err := kernels.AddPair(dAPrev, dZ, dInputTime, rows*hidden); err != nil {
			return fail(err)
		}
		dA = dAPrev
	}

	dZ1, err := grads.zeroAlloc(rows * hidden)
	if err != nil {
		return fail(err)
	}
	if err := kernels.TanhBackward(cache.inputActivation, dA, dZ1, rows*hidden); err != nil {
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
	if err := kernels.ColumnSum(dZ1, inputBias, rows, hidden); err != nil {
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
