//go:build windows && amd64

package vulkan

import (
	"fmt"
	"unsafe"
)

// These are the Vulkan 1.1 C ABI layouts on Windows/amd64. Dispatchable handles
// are uintptr; non-dispatchable handles and VkDeviceSize are uint64. Pointer
// fields use Go pointer types so pointed-to arguments stay visible to the GC.
type vkHeader struct {
	SType uint32
	Next  unsafe.Pointer
}

func header(kind uint32) vkHeader { return vkHeader{SType: kind} }

type vkApplicationInfo struct {
	vkHeader
	ApplicationName           *byte
	ApplicationVersion        uint32
	EngineName                *byte
	EngineVersion, APIVersion uint32
}
type vkInstanceCreateInfo struct {
	vkHeader
	Flags           uint32
	ApplicationInfo *vkApplicationInfo
	LayerCount      uint32
	Layers          **byte
	ExtensionCount  uint32
	Extensions      **byte
}
type vkQueueCreateInfo struct {
	vkHeader
	Flags, Family, Count uint32
	Priorities           *float32
}
type vkDeviceCreateInfo struct {
	vkHeader
	Flags, QueueCount uint32
	Queues            *vkQueueCreateInfo
	LayerCount        uint32
	Layers            **byte
	ExtensionCount    uint32
	Extensions        **byte
	Features          unsafe.Pointer
}
type vkQueueFamily struct {
	Flags, Count, TimestampBits uint32
	Granularity                 [3]uint32
}
type vkLimitsPrefix struct {
	ImageDimensions                                                           [5]uint32
	MaxTexelBufferElements, MaxUniformBufferRange, MaxStorageBufferRange      uint32
	MaxPushConstantsSize, MaxMemoryAllocationCount, MaxSamplerAllocationCount uint32
	BufferImageGranularity, SparseAddressSpaceSize                            uint64
	BeforeCompute                                                             [38]uint32
	MaxComputeSharedMemorySize                                                uint32
	MaxComputeWorkGroupCount                                                  [3]uint32
	MaxComputeWorkGroupInvocations                                            uint32
	MaxComputeWorkGroupSize                                                   [3]uint32
}
type vkPhysicalProperties struct {
	APIVersion, DriverVersion, VendorID, DeviceID, DeviceType uint32
	DeviceName                                                [256]byte
	UUID                                                      [16]byte
	Limits                                                    vkLimitsPrefix
	// Space for the rest of VkPhysicalDeviceLimits and sparse properties. The
	// prefix offsets are checked by tests; vkGetPhysicalDeviceProperties writes
	// only the fixed core struct, not future extension structs.
	Tail [1024]byte
}
type vkMemoryType struct{ Flags, Heap uint32 }
type vkMemoryHeap struct {
	Size  uint64
	Flags uint32
}
type vkMemoryProperties struct {
	TypeCount uint32
	Types     [32]vkMemoryType
	HeapCount uint32
	Heaps     [16]vkMemoryHeap
}
type vkBufferCreateInfo struct {
	vkHeader
	Flags                           uint32
	Size                            uint64
	Usage, SharingMode, FamilyCount uint32
	Families                        *uint32
}
type vkMemoryRequirements struct {
	Size, Alignment uint64
	TypeBits        uint32
}
type vkMemoryAllocateInfo struct {
	vkHeader
	Size      uint64
	TypeIndex uint32
}
type vkCommandPoolCreateInfo struct {
	vkHeader
	Flags, Family uint32
}
type vkCommandBufferAllocateInfo struct {
	vkHeader
	Pool         uint64
	Level, Count uint32
}
type vkCommandBufferBeginInfo struct {
	vkHeader
	Flags       uint32
	Inheritance unsafe.Pointer
}
type vkFenceCreateInfo struct {
	vkHeader
	Flags uint32
}
type vkSubmitInfo struct {
	vkHeader
	WaitCount        uint32
	WaitSemaphores   *uint64
	WaitStageMask    *uint32
	CommandCount     uint32
	Commands         *uintptr
	SignalCount      uint32
	SignalSemaphores *uint64
}
type vkBufferBarrier struct {
	vkHeader
	SrcAccess, DstAccess, SrcFamily, DstFamily uint32
	Buffer, Offset, Size                       uint64
}
type vkBufferCopy struct{ SrcOffset, DstOffset, Size uint64 }
type vkDescriptorBinding struct {
	Binding, Type, Count, Stage uint32
	ImmutableSamplers           *uint64
}
type vkDescriptorLayoutCreateInfo struct {
	vkHeader
	Flags, BindingCount uint32
	Bindings            *vkDescriptorBinding
}
type vkDescriptorPoolSize struct{ Type, Count uint32 }
type vkDescriptorPoolCreateInfo struct {
	vkHeader
	Flags, MaxSets, SizeCount uint32
	Sizes                     *vkDescriptorPoolSize
}
type vkDescriptorAllocateInfo struct {
	vkHeader
	Pool    uint64
	Count   uint32
	Layouts *uint64
}
type vkDescriptorBufferInfo struct{ Buffer, Offset, Range uint64 }
type vkDescriptorWrite struct {
	vkHeader
	Set                                uint64
	Binding, ArrayElement, Count, Type uint32
	ImageInfo                          unsafe.Pointer
	BufferInfo                         *vkDescriptorBufferInfo
	TexelBufferView                    *uint64
}
type vkPushRange struct{ Stage, Offset, Size uint32 }
type vkPipelineLayoutCreateInfo struct {
	vkHeader
	Flags, SetLayoutCount uint32
	SetLayouts            *uint64
	PushCount             uint32
	PushRanges            *vkPushRange
}
type vkShaderCreateInfo struct {
	vkHeader
	Flags    uint32
	CodeSize uintptr
	Code     *uint32
}
type vkShaderStage struct {
	vkHeader
	Flags, Stage   uint32
	Module         uint64
	Name           *byte
	Specialization unsafe.Pointer
}
type vkComputePipelineCreateInfo struct {
	vkHeader
	Flags                uint32
	Stage                vkShaderStage
	Layout, BasePipeline uint64
	BaseIndex            int32
}
type vkPointwisePush struct {
	N, Kind uint32
	Scalar  float32
}

type vkSpecializationMapEntry struct {
	ID, Offset uint32
	Size       uintptr
}
type vkSpecializationInfo struct {
	Count    uint32
	Entries  *vkSpecializationMapEntry
	DataSize uintptr
	Data     *uint32
}

// Refuse to call native functions if a compiler/platform layout changes. These
// sizes and offsets follow vulkan_core.h for the Windows x64 ABI.
func checkABI() error {
	for _, check := range []struct {
		name      string
		got, want uintptr
	}{
		{"header", unsafe.Sizeof(vkHeader{}), 16},
		{"application", unsafe.Sizeof(vkApplicationInfo{}), 48},
		{"instance", unsafe.Sizeof(vkInstanceCreateInfo{}), 64},
		{"queue", unsafe.Sizeof(vkQueueCreateInfo{}), 40},
		{"device", unsafe.Sizeof(vkDeviceCreateInfo{}), 72},
		{"queue family", unsafe.Sizeof(vkQueueFamily{}), 24},
		{"limits offset", unsafe.Offsetof(vkPhysicalProperties{}.Limits), 296},
		{"storage limit offset", unsafe.Offsetof(vkLimitsPrefix{}.MaxStorageBufferRange), 28},
		{"compute limit offset", unsafe.Offsetof(vkLimitsPrefix{}.MaxComputeWorkGroupCount), 220},
		{"memory properties", unsafe.Sizeof(vkMemoryProperties{}), 520},
		{"memory requirements", unsafe.Sizeof(vkMemoryRequirements{}), 24},
		{"buffer create", unsafe.Sizeof(vkBufferCreateInfo{}), 56},
		{"memory allocate", unsafe.Sizeof(vkMemoryAllocateInfo{}), 32},
		{"command pool", unsafe.Sizeof(vkCommandPoolCreateInfo{}), 24},
		{"command allocate", unsafe.Sizeof(vkCommandBufferAllocateInfo{}), 32},
		{"command begin", unsafe.Sizeof(vkCommandBufferBeginInfo{}), 32},
		{"submit", unsafe.Sizeof(vkSubmitInfo{}), 72},
		{"barrier", unsafe.Sizeof(vkBufferBarrier{}), 56},
		{"binding", unsafe.Sizeof(vkDescriptorBinding{}), 24},
		{"descriptor layout", unsafe.Sizeof(vkDescriptorLayoutCreateInfo{}), 32},
		{"descriptor pool", unsafe.Sizeof(vkDescriptorPoolCreateInfo{}), 40},
		{"descriptor allocate", unsafe.Sizeof(vkDescriptorAllocateInfo{}), 40},
		{"descriptor write", unsafe.Sizeof(vkDescriptorWrite{}), 64},
		{"pipeline layout", unsafe.Sizeof(vkPipelineLayoutCreateInfo{}), 48},
		{"shader", unsafe.Sizeof(vkShaderCreateInfo{}), 40},
		{"shader stage", unsafe.Sizeof(vkShaderStage{}), 48},
		{"compute pipeline", unsafe.Sizeof(vkComputePipelineCreateInfo{}), 96},
		{"push constants", unsafe.Sizeof(vkPointwisePush{}), 12},
		{"view push constants", unsafe.Sizeof(viewParams{}), 64},
		{"gemm push constants", unsafe.Sizeof(gemmParams{}), 100},
		{"reduction push constants", unsafe.Sizeof(reduceParams{}), 28},
		{"conv push constants", unsafe.Sizeof(convParams{}), 40},
		{"specialization entry", unsafe.Sizeof(vkSpecializationMapEntry{}), 16},
		{"specialization info", unsafe.Sizeof(vkSpecializationInfo{}), 32},
	} {
		if check.got != check.want {
			return fmt.Errorf("vulkan: ABI mismatch for %s: %d != %d", check.name, check.got, check.want)
		}
	}
	return nil
}
