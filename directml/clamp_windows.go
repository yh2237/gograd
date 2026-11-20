//go:build windows && amd64

package directml

import (
	_ "embed"
	"unsafe"
)

//go:generate fxc /nologo /T rootsig_1_0 /E ClampRoot /Fo shaders/clamp.rootsig shaders/clamp.hlsl
//go:generate fxc /nologo /T cs_5_1 /E main /O3 /Qstrip_reflect /Qstrip_rootsignature /Fo shaders/clamp.cso shaders/clamp.hlsl

//go:embed shaders/clamp.rootsig
var clampRootSignature []byte

//go:embed shaders/clamp.cso
var clampShader []byte

var (
	iidRootSignature = iid("c54a6b66-72df-4ee8-8be5-a946a1429214")
	iidPipelineState = iid("765a30f3-f624-4c6f-a828-ace948622445")
)

type computePipelineDesc struct {
	Root         *object
	Bytecode     *byte
	BytecodeSize uintptr
	NodeMask     uint32
	Cached       unsafe.Pointer
	CachedSize   uintptr
	Flags        uint32
}

func (d *nativeDriver) initClamp() error {
	if d.clampPipeline != nil {
		return nil
	}
	var root, pipeline *object
	result := call(d.device, 16, 0, uintptr(unsafe.Pointer(&clampRootSignature[0])), uintptr(len(clampRootSignature)),
		uintptr(unsafe.Pointer(iidRootSignature)), uintptr(unsafe.Pointer(&root)))
	if err := hresult("CreateRootSignature", result); err != nil {
		return err
	}
	desc := computePipelineDesc{Root: root, Bytecode: &clampShader[0], BytecodeSize: uintptr(len(clampShader))}
	result = call(d.device, 11, uintptr(unsafe.Pointer(&desc)), uintptr(unsafe.Pointer(iidPipelineState)), uintptr(unsafe.Pointer(&pipeline)))
	if err := hresult("CreateComputePipelineState", result); err != nil {
		release(root)
		return err
	}
	d.clampRoot, d.clampPipeline = root, pipeline
	return nil
}

func (d *nativeDriver) recordClamp(key opKey, out, in *allocation) {
	const threads, maxGroups = 256, 65535
	groups := (key.n + threads - 1) / threads
	groupsX := min(groups, maxGroups)
	groupsY := (groups + groupsX - 1) / groupsX
	args := [4]uint32{uint32(key.n), key.scalar, key.bias, uint32(groupsX * threads)}
	call(d.command, 25, uintptr(unsafe.Pointer(d.clampPipeline)))
	call(d.command, 29, uintptr(unsafe.Pointer(d.clampRoot)))
	call(d.command, 41, 0, call(in.native.(*object), 11))
	call(d.command, 41, 1, call(out.native.(*object), 11))
	call(d.command, 35, 2, uintptr(len(args)), uintptr(unsafe.Pointer(&args[0])), 0)
	call(d.command, 14, uintptr(groupsX), uintptr(groupsY), 1)
}
