//go:build windows && amd64

package directml

import (
	_ "embed"
	"math"
	"math/bits"
	"unsafe"
)

//go:generate fxc /nologo /T rootsig_1_0 /E UnaryRoot /Fo shaders/unary.rootsig shaders/unary.hlsl
//go:generate fxc /nologo /T cs_5_1 /E clampMain /O3 /Qstrip_reflect /Qstrip_rootsignature /Fo shaders/clamp.cso shaders/unary.hlsl
//go:generate fxc /nologo /T cs_5_1 /E powMain /O3 /Qstrip_reflect /Qstrip_rootsignature /Fo shaders/pow.cso shaders/unary.hlsl
//go:generate fxc /nologo /T cs_5_1 /E logSoftmaxMain /O3 /Qstrip_reflect /Qstrip_rootsignature /Fo shaders/logsoftmax.cso shaders/unary.hlsl

//go:embed shaders/unary.rootsig
var unaryRootSignature []byte

//go:embed shaders/clamp.cso
var clampShader []byte

//go:embed shaders/pow.cso
var powShader []byte

//go:embed shaders/logsoftmax.cso
var logSoftmaxShader []byte

var unaryShaderBytecode = [...][]byte{clampShader, powShader, logSoftmaxShader}

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

func (d *nativeDriver) initUnaryShader(kind uint32) error {
	index := kind - opClamp
	if d.unaryPipelines[index] != nil {
		return nil
	}
	if d.unaryRoot == nil {
		var root *object
		result := call(d.device, 16, 0, uintptr(unsafe.Pointer(&unaryRootSignature[0])), uintptr(len(unaryRootSignature)),
			uintptr(unsafe.Pointer(iidRootSignature)), uintptr(unsafe.Pointer(&root)))
		if err := hresult("CreateRootSignature", result); err != nil {
			return err
		}
		d.unaryRoot = root
	}
	bytecode := unaryShaderBytecode[index]
	desc := computePipelineDesc{Root: d.unaryRoot, Bytecode: &bytecode[0], BytecodeSize: uintptr(len(bytecode))}
	var pipeline *object
	result := call(d.device, 11, uintptr(unsafe.Pointer(&desc)), uintptr(unsafe.Pointer(iidPipelineState)), uintptr(unsafe.Pointer(&pipeline)))
	if err := hresult("CreateComputePipelineState", result); err != nil {
		return err
	}
	d.unaryPipelines[index] = pipeline
	return nil
}

func (d *nativeDriver) recordUnaryShader(key opKey, out, in *allocation) {
	const threads, maxGroups = 256, 65535
	items, itemsPerGroup := key.n, threads
	args := [6]uint32{uint32(key.n), key.scalar, key.bias}
	if key.kind == opLogSoftmax {
		lanesPerRow := 1
		if key.bias > 8 {
			lanesPerRow = 1 << min(bits.Len32(key.bias-1), 8)
		}
		items, itemsPerGroup = int(key.scalar), threads/lanesPerRow
		args[4] = uint32(lanesPerRow)
	}
	groups := (items + itemsPerGroup - 1) / itemsPerGroup
	groupsX := min(groups, maxGroups)
	groupsY := (groups + groupsX - 1) / groupsX
	args[3] = uint32(groupsX * itemsPerGroup)
	if key.kind == opPow {
		args[4], args[5] = powBounds(float64(math.Float32frombits(key.scalar)))
	}
	call(d.command, 25, uintptr(unsafe.Pointer(d.unaryPipelines[key.kind-opClamp])))
	call(d.command, 29, uintptr(unsafe.Pointer(d.unaryRoot)))
	call(d.command, 41, 0, call(in.native.(*object), 11))
	call(d.command, 41, 1, call(out.native.(*object), 11))
	call(d.command, 35, 2, uintptr(len(args)), uintptr(unsafe.Pointer(&args[0])), 0)
	call(d.command, 14, uintptr(groupsX), uintptr(groupsY), 1)
}

func powBounds(exponent float64) (lower, upper uint32) {
	magnitude := math.Abs(exponent)
	if !(magnitude >= 0x1p-126 && magnitude <= math.MaxFloat32) {
		return 0, 0
	}
	const roundsToZero = 0x1p-150
	const roundsToInfinity = float64(math.MaxFloat32) + 0x1p103
	const infinity = 0x7f800000
	if exponent > 0 {
		return powCutoff(exponent, roundsToZero, 1), powCutoff(exponent, roundsToInfinity, infinity)
	}
	return powCutoff(exponent, roundsToInfinity, infinity), powCutoff(exponent, roundsToZero, 1)
}

func powCutoff(exponent, limit float64, resultBits uint32) uint32 {
	const infinity = 0x7f800000
	bits := math.Float32bits(float32(math.Pow(limit, 1/exponent)))
	reached := func(bits uint32) bool {
		base := float64(math.Float32frombits(bits))
		result := math.Float32bits(float32(math.Pow(base, exponent)))
		return (result >= resultBits) == (exponent > 0)
	}
	// The estimate rounds to float32. Find the first input past the cutoff.
	for bits > 0 && reached(bits-1) {
		bits--
	}
	for bits < infinity && !reached(bits) {
		bits++
	}
	return bits
}
