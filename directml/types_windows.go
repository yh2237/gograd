//go:build windows && amd64

package directml

import (
	"fmt"
	"unsafe"
)

// Layouts and COM slots follow Windows SDK d3d12.h, dxgi.h and DirectML.h.
// In particular, COM methods returning structs use a hidden result pointer
// immediately after this, even for an eight-byte descriptor handle.
type adapterDesc struct {
	Name                             [128]uint16
	Vendor, Device, Subsys, Revision uint32
	Video, System, Shared            uint64
	LUID                             uint64
	Flags                            uint32
}
type queueDesc struct {
	Type        uint32
	Priority    int32
	Flags, Node uint32
}
type heapProperties struct{ Type, CPUPage, MemoryPool, CreationNode, VisibleNode uint32 }
type resourceDesc struct {
	Dimension                                         uint32
	Alignment, Width                                  uint64
	Height                                            uint32
	Depth, Mips                                       uint16
	Format, SampleCount, SampleQuality, Layout, Flags uint32
}
type resourceBarrier struct {
	Type, Flags                         uint32
	Resource                            *object
	Subresource, Before, After, Padding uint32
}
type descriptorHeapDesc struct{ Type, Count, Flags, Node uint32 }
type memoryRange struct{ Begin, End uintptr }
type resourceAllocationInfo struct{ Size, Alignment uint64 }
type tensorDesc struct {
	Type uint32
	Desc *bufferTensorDesc
}
type bufferTensorDesc struct {
	DataType, Flags, Dimensions uint32
	Sizes, Strides              *uint32
	Bytes                       uint64
	Alignment                   uint32
}
type operatorDesc struct {
	Type uint32
	Desc unsafe.Pointer
}
type binaryDesc struct{ A, B, Out *tensorDesc }
type identityDesc struct {
	In, Out   *tensorDesc
	ScaleBias *scaleBias
}
type scaleBias struct{ Scale, Bias float32 }
type gemmDesc struct {
	A, B, C, Out   *tensorDesc
	TransA, TransB uint32
	Alpha, Beta    float32
	Fused          *operatorDesc
}
type bindingProperties struct {
	Descriptors           uint32
	Temporary, Persistent uint64
}
type bindingTableDesc struct {
	Dispatchable *object
	CPU, GPU     uint64
	Count        uint32
}
type bufferBinding struct {
	Resource     *object
	Offset, Size uint64
}
type bindingDesc struct {
	Type uint32
	Desc *bufferBinding
}
type featureQuery struct {
	Count  uint32
	Levels *uint32
}

func checkABI() error {
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"GUID", unsafe.Sizeof(guid{}), 16}, {"adapter", unsafe.Sizeof(adapterDesc{}), 312},
		{"queue", unsafe.Sizeof(queueDesc{}), 16}, {"heap properties", unsafe.Sizeof(heapProperties{}), 20},
		{"resource", unsafe.Sizeof(resourceDesc{}), 56}, {"barrier", unsafe.Sizeof(resourceBarrier{}), 32},
		{"resource allocation info", unsafe.Sizeof(resourceAllocationInfo{}), 16},
		{"tensor", unsafe.Sizeof(tensorDesc{}), 16}, {"buffer tensor", unsafe.Sizeof(bufferTensorDesc{}), 48},
		{"operator", unsafe.Sizeof(operatorDesc{}), 16}, {"GEMM", unsafe.Sizeof(gemmDesc{}), 56},
		{"binding properties", unsafe.Sizeof(bindingProperties{}), 24}, {"binding table", unsafe.Sizeof(bindingTableDesc{}), 32},
		{"buffer binding", unsafe.Sizeof(bufferBinding{}), 24}, {"binding", unsafe.Sizeof(bindingDesc{}), 16},
		{"feature query", unsafe.Sizeof(featureQuery{}), 16},
	} {
		if c.got != c.want {
			return fmt.Errorf("directml: ABI mismatch for %s: %d != %d", c.name, c.got, c.want)
		}
	}
	return nil
}
