package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/kernels"
	"sync"
	"unsafe"
)

type clipPlan struct {
	ptrs             []uintptr
	sizesHost        []int32
	addresses, sizes *cuda.Buffer
}

// The clip plan caches the device addresses of parameter gradients, and the
// norm buffer holds the last gradient norm for diagnostics. Both belong to the
// OS thread that built them, so independent training runs neither invalidate
// each other's plan nor read each other's norm.
type clipState struct {
	plan *clipPlan
	norm *cuda.Buffer
}

var (
	clipMu     sync.Mutex
	clipStates = map[uintptr]*clipState{}
)

func clipStateForThread() *clipState {
	id := cuda.ThreadID()
	clipMu.Lock()
	defer clipMu.Unlock()
	state := clipStates[id]
	if state == nil {
		state = &clipState{}
		clipStates[id] = state
	}
	return state
}

func currentClipState() *clipState {
	clipMu.Lock()
	defer clipMu.Unlock()
	return clipStates[cuda.ThreadID()]
}

func getClipPlan(params []Parameter) *clipPlan {
	state := clipStateForThread()
	addresses := make([]uintptr, len(params))
	sizes := make([]int32, len(params))
	for i, p := range params {
		addresses[i] = p.Value.ensureGradGPU().Pointer()
		sizes[i] = int32(p.Value.Numel())
	}
	if old := state.plan; old != nil && len(old.ptrs) == len(addresses) {
		equal := true
		for i := range addresses {
			// Addresses alone are not enough: the size-class pool can hand a
			// freed block to a parameter with a different element count.
			if old.ptrs[i] != addresses[i] || old.sizesHost[i] != sizes[i] {
				equal = false
				break
			}
		}
		if equal {
			return old
		}
	}
	if state.plan != nil {
		state.plan.addresses.Free()
		state.plan.sizes.Free()
	}
	a := mustAlloc(len(addresses) * 2)
	s := mustAlloc(len(sizes))
	ap, sp := ptr(a), ptr(s)
	for i, address := range addresses {
		index, size, deviceAddress := int32(i), sizes[i], uint64(address)
		launch("write_clip_entry", 1, unsafe.Pointer(&ap), unsafe.Pointer(&sp), unsafe.Pointer(&index), unsafe.Pointer(&deviceAddress), unsafe.Pointer(&size))
	}
	state.plan = &clipPlan{ptrs: addresses, sizesHost: sizes, addresses: a, sizes: s}
	return state.plan
}
func clipGPU(params []Parameter, maxNorm float32) {
	state := clipStateForThread()
	if state.norm != nil {
		state.norm.Free()
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
	state.norm = norm
}
