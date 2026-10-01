// Package gputcn runs the frame intonation TCN forward pass on the GPU with
// the cuda package. Tensors are row-major float32 device buffers and the
// layout follows the PyTorch model: transpose to channel-major for each
// convolution and back for the linear layers.
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

// Forward runs the model on x [batch,time,inputs] and returns [batch,time].
func (m *Model) Forward(x []float32, batch, time int) ([]float32, error) {
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

	var buffers []*cuda.Buffer
	release := func() {
		for _, buffer := range buffers {
			buffer.Free()
		}
	}
	defer release()
	allocate := func(count int) (*cuda.Buffer, error) {
		buffer, err := cuda.Alloc(count * 4)
		if err == nil {
			buffers = append(buffers, buffer)
		}
		return buffer, err
	}

	rows := batch * time
	input, err := cuda.Alloc(rows * m.Inputs * 4)
	if err != nil {
		return nil, err
	}
	buffers = append(buffers, input)
	if err := input.CopyFromHost(floatsToBytes(x)); err != nil {
		return nil, err
	}

	state, err := allocate(rows * m.Hidden)
	if err != nil {
		return nil, err
	}
	if err := blas.SgemmRowMajorNT(rows, m.Hidden, m.Inputs, 1, input.Pointer(), m.Inputs, m.InputWeight.Pointer(), m.Inputs, 0, state.Pointer(), m.Hidden); err != nil {
		return nil, err
	}
	if err := cuda.BiasColumnsTanh(state, m.InputBias, rows, m.Hidden); err != nil {
		return nil, err
	}

	for index := range m.Layers {
		layer := &m.Layers[index]
		channelMajor, err := allocate(batch * m.Hidden * time)
		if err != nil {
			return nil, err
		}
		if err := cuda.Transpose12(state, channelMajor, batch, time, m.Hidden); err != nil {
			return nil, err
		}
		convolved, err := allocate(batch * m.Hidden * time)
		if err != nil {
			return nil, err
		}
		if err := cuda.Conv1dForward(blas, channelMajor, layer.Weight, layer.Bias, convolved, batch, m.Hidden, time, m.Hidden, 3, layer.Dilation); err != nil {
			return nil, err
		}
		convolvedTime, err := allocate(batch * time * m.Hidden)
		if err != nil {
			return nil, err
		}
		if err := cuda.Transpose12(convolved, convolvedTime, batch, m.Hidden, time); err != nil {
			return nil, err
		}
		next, err := allocate(batch * time * m.Hidden)
		if err != nil {
			return nil, err
		}
		if err := cuda.AddTanh(state, convolvedTime, next, rows*m.Hidden); err != nil {
			return nil, err
		}
		state = next
	}

	output, err := allocate(rows)
	if err != nil {
		return nil, err
	}
	if err := blas.SgemmRowMajorNT(rows, 1, m.Hidden, 1, state.Pointer(), m.Hidden, m.OutputWeight.Pointer(), m.Hidden, 0, output.Pointer(), 1); err != nil {
		return nil, err
	}
	if err := cuda.AddBiasColumns(output, m.OutputBias, rows, 1); err != nil {
		return nil, err
	}
	if err := cuda.Synchronize(); err != nil {
		return nil, err
	}
	result := make([]float32, rows)
	if err := output.CopyToHost(floatsToBytes(result)); err != nil {
		return nil, err
	}
	return result, nil
}

func floatsToBytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
