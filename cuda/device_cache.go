package cuda

import (
	"runtime"
	"sync"
)

// DeviceCache lazily builds context-owned resources once per CUDA device.
// Callers launching device work must keep their OS thread pinned throughout
// lookup and use, just as they do for other current-device CUDA operations.
// Build failures are remembered independently for each device.
type DeviceCache[T any] struct {
	mu      sync.Mutex
	entries map[int]*deviceCacheEntry[T]
}

type deviceCacheEntry[T any] struct {
	once  sync.Once
	value T
	err   error
}

func (c *DeviceCache[T]) Get(build func() (T, error)) (T, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	device, err := CurrentDevice()
	if err != nil {
		var zero T
		return zero, err
	}
	c.mu.Lock()
	if c.entries == nil {
		c.entries = make(map[int]*deviceCacheEntry[T])
	}
	entry := c.entries[device]
	if entry == nil {
		entry = &deviceCacheEntry[T]{}
		c.entries[device] = entry
	}
	c.mu.Unlock()
	entry.once.Do(func() { entry.value, entry.err = build() })
	return entry.value, entry.err
}

// KernelCache owns one compiled module and its named functions per device.
type KernelCache struct {
	devices DeviceCache[map[string]*Kernel]
}

func (c *KernelCache) Get(source string, names ...string) (map[string]*Kernel, error) {
	return c.devices.Get(func() (map[string]*Kernel, error) {
		program, err := Compile(source)
		if err != nil {
			return nil, err
		}
		functions := make(map[string]*Kernel, len(names))
		for _, name := range names {
			function, err := program.Function(name)
			if err != nil {
				_ = program.Close()
				return nil, err
			}
			functions[name] = function
		}
		// Kernels retain the module handle; successful modules live with the cache.
		return functions, nil
	})
}
