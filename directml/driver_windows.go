//go:build windows && amd64

package directml

import (
	"fmt"
	"os"
	"runtime"
	"syscall"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

const (
	stateUAV           = 0x8
	stateCopyDest      = 0x400
	stateCopySource    = 0x800
	stateGenericRead   = 0xac3
	maxCachedOperators = 64
	operatorCacheBytes = 256 << 20
)

type nativeDriver struct {
	dlls                                                    []*syscall.DLL
	device, dml, queue, allocator, command, fence, recorder *object
	serial                                                  uint64
	properties                                              Info
	operators                                               map[opKey]*compiled
	order                                                   []opKey
}

func openDriver() (_ driver, err error) {
	if err := checkABI(); err != nil {
		return nil, err
	}
	d := &nativeDriver{}
	defer func() {
		if err != nil {
			d.close()
		}
	}()
	for _, name := range []string{"dxgi.dll", "d3d12.dll", "DirectML.dll"} {
		dll, e := systemDLL(name)
		if e != nil {
			return nil, e
		}
		d.dlls = append(d.dlls, dll)
	}
	createFactory, e := d.dlls[0].FindProc("CreateDXGIFactory1")
	if e != nil {
		return nil, e
	}
	createD3D, e := d.dlls[1].FindProc("D3D12CreateDevice")
	if e != nil {
		return nil, e
	}
	createDML, e := d.dlls[2].FindProc("DMLCreateDevice")
	if e != nil {
		return nil, e
	}
	var factory *object
	r, _, _ := createFactory.Call(uintptr(unsafe.Pointer(iidFactory)), uintptr(unsafe.Pointer(&factory)))
	if err := hresult("CreateDXGIFactory1", r); err != nil {
		return nil, err
	}
	defer release(factory)
	var best uint64
	// GOGRAD_DIRECTML_SOFTWARE selects the software rasterizer instead of a
	// hardware adapter. It exists to reproduce the behaviour of machines
	// without a GPU, which is what the CI runners use.
	software := os.Getenv("GOGRAD_DIRECTML_SOFTWARE") == "1"
	for i := 0; i < 256; i++ {
		var adapter *object
		r := call(factory, 12, uintptr(i), uintptr(unsafe.Pointer(&adapter)))
		if uint32(r) == 0x887a0002 {
			break
		}
		if err := hresult("EnumAdapters1", r); err != nil {
			return nil, err
		}
		var desc adapterDesc
		if err := hresult("GetDesc1", call(adapter, 10, uintptr(unsafe.Pointer(&desc)))); err != nil {
			release(adapter)
			return nil, err
		}
		if (desc.Flags&2 != 0) != software {
			release(adapter)
			continue
		} // Never use WARP/software unless it was requested explicitly.
		var dev, ml *object
		r, _, _ = createD3D.Call(uintptr(unsafe.Pointer(adapter)), 0xb000, uintptr(unsafe.Pointer(iidD3DDevice)), uintptr(unsafe.Pointer(&dev)))
		release(adapter)
		if int32(r) < 0 {
			release(dev)
			continue
		}
		r, _, _ = createDML.Call(uintptr(unsafe.Pointer(dev)), 0, uintptr(unsafe.Pointer(iidDMLDevice)), uintptr(unsafe.Pointer(&ml)))
		if int32(r) < 0 {
			release(ml)
			release(dev)
			continue
		}
		score := desc.Video + 1
		if d.device == nil || score > best {
			release(d.dml)
			release(d.device)
			d.device, d.dml, best = dev, ml, score
			d.properties = Info{Name: syscall.UTF16ToString(desc.Name[:]), VendorID: desc.Vendor, DeviceID: desc.Device, DedicatedVideoMemory: desc.Video, FeatureLevel: 0x1000}
		} else {
			release(ml)
			release(dev)
		}
	}
	if d.device == nil {
		return nil, fmt.Errorf("%w: no DirectML-capable hardware adapter", gpu.ErrUnavailable)
	}
	dataType, supported := uint32(1), uint32(0)
	if err := hresult("CheckFeatureSupport(FLOAT32)", call(d.dml, 7, 0, 4, uintptr(unsafe.Pointer(&dataType)), 4, uintptr(unsafe.Pointer(&supported)))); err != nil {
		return nil, err
	}
	if supported == 0 {
		return nil, fmt.Errorf("%w: DirectML float32 unsupported", gpu.ErrUnavailable)
	}
	levels := [...]uint32{0x6400, 0x6300, 0x6200, 0x6100, 0x6000, 0x5200, 0x5100, 0x5000, 0x4100, 0x4000, 0x3100, 0x3000, 0x2100, 0x2000, 0x1000}
	query := featureQuery{Count: uint32(len(levels)), Levels: &levels[0]}
	var level uint32
	if int32(call(d.dml, 7, 1, unsafe.Sizeof(query), uintptr(unsafe.Pointer(&query)), 4, uintptr(unsafe.Pointer(&level)))) >= 0 && level >= 0x1000 {
		d.properties.FeatureLevel = level
	}
	q := queueDesc{} // DIRECT lists support DirectML and buffer copies.
	if err := hresult("CreateCommandQueue", call(d.device, 8, uintptr(unsafe.Pointer(&q)), uintptr(unsafe.Pointer(iidQueue)), uintptr(unsafe.Pointer(&d.queue)))); err != nil {
		return nil, err
	}
	if err := hresult("CreateCommandAllocator", call(d.device, 9, 0, uintptr(unsafe.Pointer(iidAllocator)), uintptr(unsafe.Pointer(&d.allocator)))); err != nil {
		return nil, err
	}
	if err := hresult("CreateCommandList", call(d.device, 12, 0, 0, uintptr(unsafe.Pointer(d.allocator)), 0, uintptr(unsafe.Pointer(iidCommand)), uintptr(unsafe.Pointer(&d.command)))); err != nil {
		return nil, err
	}
	if err := hresult("Close", call(d.command, 9)); err != nil {
		return nil, err
	}
	if err := hresult("CreateFence", call(d.device, 36, 0, 0, uintptr(unsafe.Pointer(iidFence)), uintptr(unsafe.Pointer(&d.fence)))); err != nil {
		return nil, err
	}
	if err := hresult("CreateCommandRecorder", call(d.dml, 11, uintptr(unsafe.Pointer(iidRecorder)), uintptr(unsafe.Pointer(&d.recorder)))); err != nil {
		return nil, err
	}
	return d, nil
}

func (d *nativeDriver) info() Info { return d.properties }
func (d *nativeDriver) close() {
	d.releaseOperators()
	for _, o := range []*object{d.recorder, d.fence, d.command, d.allocator, d.queue, d.dml, d.device} {
		release(o)
	}
	for i := len(d.dlls) - 1; i >= 0; i-- {
		d.dlls[i].Release()
	}
	d.dlls = nil
}

func (d *nativeDriver) alloc(size uint64, heap uint32) (*allocation, error) {
	size = (size + 255) &^ 255
	props := heapProperties{Type: heap, CreationNode: 1, VisibleNode: 1}
	desc := resourceDesc{Dimension: 1, Width: size, Height: 1, Depth: 1, Mips: 1, SampleCount: 1, Layout: 1}
	state := uint32(0)
	if heap == heapDefault {
		desc.Flags = 4
	} else if heap == heapUpload {
		state = stateGenericRead
	} else {
		state = stateCopyDest
	}
	var resource *object
	if err := hresult("CreateCommittedResource", call(d.device, 27, uintptr(unsafe.Pointer(&props)), 0, uintptr(unsafe.Pointer(&desc)), uintptr(state), 0, uintptr(unsafe.Pointer(iidResource)), uintptr(unsafe.Pointer(&resource)))); err != nil {
		return nil, err
	}
	// Small committed buffers still reserve at least a D3D12 placement page.
	// Charge that allocation, not just Width, against the pool/cache limits.
	var info resourceAllocationInfo
	call(d.device, 25, uintptr(unsafe.Pointer(&info)), 0, 1, uintptr(unsafe.Pointer(&desc)))
	if info.Size < size || info.Size == ^uint64(0) {
		release(resource)
		return nil, fmt.Errorf("directml: invalid resource allocation info")
	}
	return &allocation{native: resource, size: size, allocated: info.Size, heap: heap}, nil
}
func (d *nativeDriver) free(v *allocation) { release(v.native.(*object)) }
func (d *nativeDriver) write(v *allocation, values []float32) error {
	var mapped unsafe.Pointer
	empty := memoryRange{}
	r := v.native.(*object)
	if err := hresult("Map(upload)", call(r, 8, 0, uintptr(unsafe.Pointer(&empty)), uintptr(unsafe.Pointer(&mapped)))); err != nil {
		return err
	}
	if len(values) == 0 {
		clear(unsafe.Slice((*byte)(mapped), int(v.size)))
	} else {
		copy(unsafe.Slice((*float32)(mapped), len(values)), values)
	}
	written := memoryRange{End: uintptr(v.size)}
	call(r, 9, 0, uintptr(unsafe.Pointer(&written)))
	return nil
}
func (d *nativeDriver) read(v *allocation, values []float32) error {
	var mapped unsafe.Pointer
	span := memoryRange{End: uintptr(len(values)) * 4}
	r := v.native.(*object)
	if err := hresult("Map(readback)", call(r, 8, 0, uintptr(unsafe.Pointer(&span)), uintptr(unsafe.Pointer(&mapped)))); err != nil {
		return err
	}
	copy(values, unsafe.Slice((*float32)(mapped), len(values)))
	empty := memoryRange{}
	call(r, 9, 0, uintptr(unsafe.Pointer(&empty)))
	return nil
}
func (d *nativeDriver) transition(v *allocation, before, after uint32) {
	if v.heap != heapDefault {
		return
	}
	barrier := resourceBarrier{Resource: v.native.(*object), Subresource: ^uint32(0), Before: before, After: after}
	call(d.command, 26, 1, uintptr(unsafe.Pointer(&barrier)))
}
func (d *nativeDriver) uavBarrier() {
	barrier := resourceBarrier{Type: 2}
	call(d.command, 26, 1, uintptr(unsafe.Pointer(&barrier)))
}

func (d *nativeDriver) submit(record func()) error {
	if err := hresult("Reset(allocator)", call(d.allocator, 8)); err != nil {
		return err
	}
	if err := hresult("Reset(command)", call(d.command, 10, uintptr(unsafe.Pointer(d.allocator)), 0)); err != nil {
		return err
	}
	record()
	if err := hresult("Close(command)", call(d.command, 9)); err != nil {
		return err
	}
	call(d.queue, 10, 1, uintptr(unsafe.Pointer(&d.command)))
	d.serial++
	// Never return buffers to a pool while a submitted list may still use them.
	// If Signal fails, retry until the queue can be fenced or device removal
	// permits destruction. A host event-allocation error falls back to polling.
	var signalError error
	for {
		err := hresult("Signal", call(d.queue, 14, uintptr(unsafe.Pointer(d.fence)), uintptr(d.serial)))
		if err == nil {
			break
		}
		signalError = err
		if int32(call(d.device, 37)) < 0 {
			return err
		}
		runtime.Gosched()
	}
	waitError := hresult("SetEventOnCompletion", call(d.fence, 9, uintptr(d.serial), 0))
	for {
		done := uint64(call(d.fence, 8))
		if done == ^uint64(0) {
			return fmt.Errorf("directml: device removed while waiting for fence")
		}
		if done >= d.serial {
			break
		}
		if int32(call(d.device, 37)) < 0 {
			return fmt.Errorf("directml: device removed while waiting for fence")
		}
		runtime.Gosched()
	}
	if signalError != nil {
		return signalError
	}
	return waitError
}
func (d *nativeDriver) copy(out, src *allocation, size uint64) error {
	return d.submit(func() {
		d.transition(src, 0, stateCopySource)
		d.transition(out, 0, stateCopyDest)
		call(d.command, 15, uintptr(unsafe.Pointer(out.native.(*object))), 0, uintptr(unsafe.Pointer(src.native.(*object))), 0, uintptr(size))
		d.transition(src, stateCopySource, 0)
		d.transition(out, stateCopyDest, 0)
	})
}
