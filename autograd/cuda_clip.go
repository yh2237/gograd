package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"unsafe"
)

type clipPlan struct {
	ptrs             []uintptr
	addresses, sizes *cuda.Buffer
}

var currentClipPlan *clipPlan

func getClipPlan(params []Parameter) *clipPlan {
	addresses := make([]uintptr, len(params))
	sizes := make([]int32, len(params))
	for i, p := range params {
		addresses[i] = p.Value.ensureGradGPU().Pointer()
		sizes[i] = int32(p.Value.Numel())
	}
	if old := currentClipPlan; old != nil && len(old.ptrs) == len(addresses) {
		equal := true
		for i := range addresses {
			if old.ptrs[i] != addresses[i] {
				equal = false
				break
			}
		}
		if equal {
			return old
		}
	}
	if currentClipPlan != nil {
		currentClipPlan.addresses.Free()
		currentClipPlan.sizes.Free()
	}
	a := mustAlloc(len(addresses) * 2)
	s := mustAlloc(len(sizes))
	ap, sp := ptr(a), ptr(s)
	for i, address := range addresses {
		index, size, deviceAddress := int32(i), sizes[i], uint64(address)
		launch("write_clip_entry", 1, unsafe.Pointer(&ap), unsafe.Pointer(&sp), unsafe.Pointer(&index), unsafe.Pointer(&deviceAddress), unsafe.Pointer(&size))
	}
	currentClipPlan = &clipPlan{ptrs: addresses, addresses: a, sizes: s}
	return currentClipPlan
}
func clipGPU(params []Parameter, maxNorm float32) {
	if lastNormGPU != nil {
		lastNormGPU.Free()
	}
	norm := mustAlloc(1)
	if e := norm.Memset(0, 4); e != nil {
		panic(e)
	}
	plan := getClipPlan(params)
	ap, sp, np := ptr(plan.addresses), ptr(plan.sizes), ptr(norm)
	count := int32(len(params))
	launch("global_norm_parts", len(params)*256, unsafe.Pointer(&ap), unsafe.Pointer(&sp), unsafe.Pointer(&np), unsafe.Pointer(&count))
	if e := kernels.SqrtScalar(norm); e != nil {
		panic(e)
	}
	if maxNorm > 0 {
		launch("scale_all_grads", len(params)*256, unsafe.Pointer(&ap), unsafe.Pointer(&sp), unsafe.Pointer(&np), unsafe.Pointer(&count), unsafe.Pointer(&maxNorm))
	}
	lastNormGPU = norm
}
