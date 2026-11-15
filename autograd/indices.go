package autograd

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// IndexBuffer stores fixed embedding indices on CUDA, so repeated forwards
// and CUDA graph capture do not upload indices each step.
type IndexBuffer struct {
	buffer  *cuda.Buffer
	storage *tensorStorage
	Count   int
}

func NewIndexBuffer(ids []int) (*IndexBuffer, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("autograd: empty index buffer")
	}
	values := make([]int32, len(ids))
	for i, v := range ids {
		// The device kernel reads int32, so a wider index would wrap silently.
		if v < 0 || v > math.MaxInt32 {
			return nil, fmt.Errorf("autograd: index out of range")
		}
		values[i] = int32(v)
	}
	b, e := cuda.Alloc(len(values) * 4)
	if e != nil {
		return nil, e
	}
	if e = b.CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)); e != nil {
		b.Free()
		return nil, e
	}
	return &IndexBuffer{buffer: b, storage: deviceStorage(b), Count: len(ids)}, nil
}
func (ids *IndexBuffer) Close() {
	if ids != nil && ids.buffer != nil {
		ids.storage.release()
		ids.buffer = nil
		ids.storage = nil
	}
}
func EmbeddingFromIndexBuffer(weight *Tensor, ids *IndexBuffer, shape []int, paddingIdx int) *Tensor {
	dispatchBackend("embedding_index_buffer", weight.Device)
	if weight.Device != tensor.CUDA || ids == nil || ids.buffer == nil || len(weight.Shape) != 2 || numel(shape) != ids.Count {
		panic("autograd: CUDA embedding indices shape")
	}
	if paddingIdx < -1 || paddingIdx >= weight.Shape[0] {
		panic("autograd: padding index")
	}
	r := gpuEmbeddingBuffer(weight.Contiguous(), ids.buffer, shape, paddingIdx, false)
	r.savedLeases = append(r.savedLeases, ids.storage.acquire())
	return r
}
