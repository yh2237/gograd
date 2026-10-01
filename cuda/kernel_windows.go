//go:build windows

package cuda

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

// kernelAPI binds NVRTC and the CUDA driver. NVRTC compiles the embedded
// kernel source to PTX at run time, so no build-time nvcc step is needed.
type kernelAPI struct {
	nvrtc  *syscall.DLL
	driver *syscall.DLL

	createProgram  *syscall.Proc
	compileProgram *syscall.Proc
	getPTXSize     *syscall.Proc
	getPTX         *syscall.Proc
	getLogSize     *syscall.Proc
	getLog         *syscall.Proc
	destroyProgram *syscall.Proc

	moduleLoadData    *syscall.Proc
	moduleGetFunction *syscall.Proc
	moduleUnload      *syscall.Proc
	launchKernel      *syscall.Proc

	streamBeginCapture *syscall.Proc
	streamEndCapture   *syscall.Proc
	graphInstantiate   *syscall.Proc
	graphLaunch        *syscall.Proc
	graphExecDestroy   *syscall.Proc
	graphDestroy       *syscall.Proc
}

var (
	kernelOnce sync.Once
	kernelInst *kernelAPI
	kernelErr  error
)

func loadKernelAPI() (*kernelAPI, error) {
	kernelOnce.Do(func() {
		nvrtcNames := []string{}
		for _, base := range []string{
			"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll",
			"nvrtc64_111_0.dll", "nvrtc64_110_0.dll", "nvrtc64_102_0.dll",
			"nvrtc64_101_0.dll", "nvrtc64_100_0.dll",
		} {
			nvrtcNames = append(nvrtcNames, libraryCandidates(base)...)
		}
		nvrtc, err := loadLibrary(nvrtcNames)
		if err != nil {
			kernelErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		driver, err := syscall.LoadDLL("nvcuda.dll")
		if err != nil {
			kernelErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		nvrtcProcs, err := findProcs(nvrtc,
			"nvrtcCreateProgram", "nvrtcCompileProgram", "nvrtcGetPTXSize",
			"nvrtcGetPTX", "nvrtcGetProgramLogSize", "nvrtcGetProgramLog",
			"nvrtcDestroyProgram",
		)
		if err != nil {
			kernelErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		driverProcs, err := findProcs(driver,
			"cuModuleLoadData", "cuModuleGetFunction", "cuModuleUnload", "cuLaunchKernel",
		)
		if err != nil {
			kernelErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		graphProcs, err := findProcsAny(driver, [][]string{
			{"cuStreamBeginCapture_v2", "cuStreamBeginCapture"},
			{"cuStreamEndCapture"},
			{"cuGraphInstantiateWithFlags", "cuGraphInstantiate"},
			{"cuGraphLaunch"},
			{"cuGraphExecDestroy"},
			{"cuGraphDestroy"},
		})
		if err != nil {
			kernelErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		kernelInst = &kernelAPI{
			nvrtc: nvrtc, driver: driver,
			createProgram: nvrtcProcs[0], compileProgram: nvrtcProcs[1],
			getPTXSize: nvrtcProcs[2], getPTX: nvrtcProcs[3],
			getLogSize: nvrtcProcs[4], getLog: nvrtcProcs[5], destroyProgram: nvrtcProcs[6],
			moduleLoadData: driverProcs[0], moduleGetFunction: driverProcs[1],
			moduleUnload: driverProcs[2], launchKernel: driverProcs[3],
			streamBeginCapture: graphProcs[0], streamEndCapture: graphProcs[1],
			graphInstantiate: graphProcs[2], graphLaunch: graphProcs[3],
			graphExecDestroy: graphProcs[4], graphDestroy: graphProcs[5],
		}
	})
	return kernelInst, kernelErr
}

func cStringBytes(text string) []byte {
	bytes := make([]byte, len(text)+1)
	copy(bytes, text)
	return bytes
}

// Kernel is a compiled device function.
type Kernel struct {
	module   uintptr
	function uintptr
	owned    bool
}

// Program is a compiled module that can resolve several functions.
type Program struct {
	module uintptr
}

// Compile builds an NVRTC program for the current device and loads it. The
// module stays loaded until Close.
func Compile(source string) (*Program, error) {
	module, err := compileModule(source)
	if err != nil {
		return nil, err
	}
	return &Program{module: module}, nil
}

// Function resolves a __global__ function from the program.
func (p *Program) Function(name string) (*Kernel, error) {
	return moduleFunction(p.module, name)
}

// Close unloads the module.
func (p *Program) Close() error {
	if p.module == 0 {
		return nil
	}
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	code, _, _ := a.moduleUnload.Call(p.module)
	p.module = 0
	return driverError("cuModuleUnload", code)
}

func compileModule(source string) (uintptr, error) {
	a, err := loadKernelAPI()
	if err != nil {
		return 0, err
	}
	if err := SetDevice(0); err == nil {
		_ = initializeContext()
	}
	device, err := CurrentDevice()
	if err != nil {
		return 0, err
	}
	major, err := DeviceAttribute(75, device)
	if err != nil {
		return 0, err
	}
	minor, err := DeviceAttribute(76, device)
	if err != nil {
		return 0, err
	}
	architecture := fmt.Sprintf("--gpu-architecture=compute_%d%d", major, minor)

	sourceBytes := cStringBytes(source)
	nameBytes := cStringBytes("gograd_kernel")
	var program uintptr
	create, _, _ := a.createProgram.Call(
		uintptr(unsafe.Pointer(&program)),
		uintptr(unsafe.Pointer(&sourceBytes[0])),
		uintptr(unsafe.Pointer(&nameBytes[0])),
		0, 0, 0,
	)
	if create != 0 {
		return 0, &Error{Op: "nvrtcCreateProgram", Code: int(create)}
	}
	defer func() {
		a.destroyProgram.Call(uintptr(unsafe.Pointer(&program)))
	}()

	optionTexts := []string{architecture}
	optionBytes := make([][]byte, len(optionTexts))
	optionPtrs := make([]uintptr, len(optionTexts))
	for i, option := range optionTexts {
		optionBytes[i] = cStringBytes(option)
		optionPtrs[i] = uintptr(unsafe.Pointer(&optionBytes[i][0]))
	}
	compile, _, _ := a.compileProgram.Call(program, uintptr(len(optionPtrs)), uintptr(unsafe.Pointer(&optionPtrs[0])))
	runtime.KeepAlive(optionBytes)
	if compile != 0 {
		return 0, &Error{Op: "nvrtcCompileProgram", Code: int(compile), Message: programLog(a, program)}
	}

	var size uintptr
	if code, _, _ := a.getPTXSize.Call(program, uintptr(unsafe.Pointer(&size))); code != 0 {
		return 0, &Error{Op: "nvrtcGetPTXSize", Code: int(code)}
	}
	ptx := make([]byte, size)
	if code, _, _ := a.getPTX.Call(program, uintptr(unsafe.Pointer(&ptx[0]))); code != 0 {
		return 0, &Error{Op: "nvrtcGetPTX", Code: int(code)}
	}

	var module uintptr
	if code, _, _ := a.moduleLoadData.Call(uintptr(unsafe.Pointer(&module)), uintptr(unsafe.Pointer(&ptx[0]))); code != 0 {
		return 0, &Error{Op: "cuModuleLoadData", Code: int(code)}
	}
	return module, nil
}

func moduleFunction(module uintptr, name string) (*Kernel, error) {
	a, err := loadKernelAPI()
	if err != nil {
		return nil, err
	}
	nameBytes := cStringBytes(name)
	var function uintptr
	code, _, _ := a.moduleGetFunction.Call(uintptr(unsafe.Pointer(&function)), module, uintptr(unsafe.Pointer(&nameBytes[0])))
	if code != 0 {
		return nil, &Error{Op: "cuModuleGetFunction", Code: int(code)}
	}
	return &Kernel{module: module, function: function}, nil
}

// LoadKernel compiles source with NVRTC for the current device and resolves
// the named __global__ function. The returned kernel owns the module and
// unloads it on Close.
func LoadKernel(source, name string) (*Kernel, error) {
	module, err := compileModule(source)
	if err != nil {
		return nil, err
	}
	kernel, err := moduleFunction(module, name)
	if err != nil {
		a, _ := loadKernelAPI()
		if a != nil {
			a.moduleUnload.Call(module)
		}
		return nil, err
	}
	kernel.owned = true
	return kernel, nil
}

func programLog(a *kernelAPI, program uintptr) string {
	var size uintptr
	if code, _, _ := a.getLogSize.Call(program, uintptr(unsafe.Pointer(&size))); code != 0 || size == 0 {
		return ""
	}
	buffer := make([]byte, size)
	if code, _, _ := a.getLog.Call(program, uintptr(unsafe.Pointer(&buffer[0]))); code != 0 {
		return ""
	}
	return strings.TrimRight(string(buffer), "\x00\n\r ")
}

// Launch runs the kernel. grid and block are {x,y,z}. args holds one pointer
// per parameter, each pointing at the argument value. stream may be nil for
// the default stream.
func (k *Kernel) Launch(grid, block [3]int, sharedMemory int, stream *Stream, args []unsafe.Pointer) error {
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	streamHandle := currentStream
	if stream != nil {
		streamHandle = stream.handle
	}
	var arguments uintptr
	if len(args) > 0 {
		arguments = uintptr(unsafe.Pointer(&args[0]))
	}
	code, _, _ := a.launchKernel.Call(
		k.function,
		uintptr(int32(grid[0])), uintptr(int32(grid[1])), uintptr(int32(grid[2])),
		uintptr(int32(block[0])), uintptr(int32(block[1])), uintptr(int32(block[2])),
		uintptr(sharedMemory), streamHandle, arguments, 0,
	)
	runtime.KeepAlive(args)
	return driverError("cuLaunchKernel", code)
}

// Close unloads the module when the kernel owns it. Kernels obtained from a
// Program share that program's module and are released by Program.Close.
func (k *Kernel) Close() error {
	if !k.owned {
		return nil
	}
	a, err := loadKernelAPI()
	if err != nil {
		return err
	}
	k.owned = false
	code, _, _ := a.moduleUnload.Call(k.module)
	k.module = 0
	k.function = 0
	return driverError("cuModuleUnload", code)
}

var driverMessages = map[int]string{
	1:   "invalid value",
	100: "no device",
	101: "invalid device",
	201: "invalid context",
	400: "invalid handle",
	401: "invalid image",
	500: "not found",
	700: "illegal address",
	701: "launch out of resources",
}

func driverError(op string, code uintptr) error {
	if code == 0 {
		return nil
	}
	return &Error{Op: op, Code: int(code), Message: driverMessages[int(code)]}
}
