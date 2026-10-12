//go:build windows

package cuda

import (
	"runtime"
	"testing"
	"unsafe"
)

// movePooledPointerLocked re-keys one cached pointer so a test can observe how
// the pool matches on device without needing two physical devices.
func movePooledPointerLocked(from, to poolKey, pointer uintptr) {
	list := poolTable[from]
	for i, candidate := range list {
		if candidate == pointer {
			list[i] = list[len(list)-1]
			list = list[:len(list)-1]
			break
		}
	}
	if len(list) == 0 {
		delete(poolTable, from)
	} else {
		poolTable[from] = list
	}
	poolTable[to] = append(poolTable[to], pointer)
	if node := poolNodes[pointer]; node != nil {
		value := node.Value.(cachedPointer)
		value.device = to.device
		node.Value = value
	}
}

func TestAllocPoolKeepsDevicesApart(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	device, err := CurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	const size = 1024
	buffer, err := Alloc(size)
	if err != nil {
		t.Fatal(err)
	}
	if buffer.Device() != device {
		t.Fatalf("buffer device %d, current device %d", buffer.Device(), device)
	}
	pointer := buffer.Pointer()
	if err := buffer.Free(); err != nil {
		t.Fatal(err)
	}
	class := poolSizeClass(size)
	home, other := poolKey{device: device, size: class}, poolKey{device: device + 1, size: class}
	// Pretend the cached block belongs to another device; Alloc must not hand it
	// back while this device is current.
	poolMu.Lock()
	movePooledPointerLocked(home, other, pointer)
	poolMu.Unlock()
	restored := false
	t.Cleanup(func() {
		poolMu.Lock()
		if !restored {
			movePooledPointerLocked(other, home, pointer)
		}
		poolMu.Unlock()
		_ = ReleasePool()
	})
	fresh, err := Alloc(size)
	if err != nil {
		t.Fatal(err)
	}
	if fresh.Pointer() == pointer {
		t.Fatal("pool reused a block cached for another device")
	}
	if err := fresh.Free(); err != nil {
		t.Fatal(err)
	}
	// Moving the block back restores reuse on the matching device.
	poolMu.Lock()
	movePooledPointerLocked(other, home, pointer)
	poolMu.Unlock()
	restored = true
	reused, err := Alloc(size)
	if err != nil {
		t.Fatal(err)
	}
	if reused.Pointer() != pointer {
		t.Fatal("pool did not reuse a block cached for the current device")
	}
	if err := reused.Free(); err != nil {
		t.Fatal(err)
	}
}

func TestBufferOperationsRestoreDevice(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	device, err := CurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	buffer, err := Alloc(64)
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Free()
	values := make([]float32, 16)
	values[3] = 7
	if err := buffer.CopyFromHost(floatsToBytes(values)); err != nil {
		t.Fatal(err)
	}
	if err := buffer.Memset(0, 4); err != nil {
		t.Fatal(err)
	}
	if err := buffer.MemsetAsync(0, 4); err != nil {
		t.Fatal(err)
	}
	if err := buffer.Free(); err != nil {
		t.Fatal(err)
	}
	after, err := CurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	if after != device {
		t.Fatalf("current device changed from %d to %d", device, after)
	}
}

func TestThreadStreamsAreIndependent(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	type result struct {
		stream uintptr
		seen   uintptr
		err    error
	}
	created := make(chan struct{}, 2)
	hold := make(chan struct{})
	run := func() result {
		var out result
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		var stream *Stream
		if err := SetDevice(0); err != nil {
			out.err = err
		} else if stream, out.err = NewStream(); out.err == nil {
			out.stream = stream.handle
			SetCurrentStream(stream)
			defer SetCurrentStream(nil)
			defer stream.Destroy()
		}
		// Announce this thread's scope, then hold it until both scopes exist, so
		// the two streams are alive at the same time instead of the driver reusing
		// one handle after the other scope closed. Both goroutines always reach
		// the barrier so a setup failure cannot deadlock the test.
		created <- struct{}{}
		<-hold
		if stream != nil {
			out.seen = CurrentStreamHandle()
		}
		return out
	}
	results := make(chan result, 2)
	go func() { results <- run() }()
	go func() { results <- run() }()
	<-created
	<-created
	close(hold)
	a, b := <-results, <-results
	if a.err != nil || b.err != nil {
		t.Fatalf("stream setup: %v %v", a.err, b.err)
	}
	if a.stream == 0 || b.stream == 0 || a.stream == b.stream {
		t.Fatalf("expected two different live streams, got %#x and %#x", a.stream, b.stream)
	}
	if a.seen != a.stream || b.seen != b.stream {
		t.Fatalf("threads saw each other's stream: %#x/%#x vs %#x/%#x", a.seen, a.stream, b.seen, b.stream)
	}
	// The calling test goroutine never selected a stream.
	if CurrentStreamHandle() != 0 || InCapture() {
		t.Fatal("thread stream state leaked into an unrelated goroutine")
	}
}

func TestCaptureRejectsNestingAndNilStream(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := SetDevice(0); err != nil {
		t.Fatal(err)
	}
	if _, err := Capture(nil, func() error { return nil }); err == nil {
		t.Fatal("capture without a stream accepted")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	var nested error
	graph, err := Capture(stream, func() error {
		_, nested = Capture(stream, func() error { return nil })
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if nested == nil {
		t.Fatal("nested capture accepted")
	}
	if InCapture() {
		t.Fatal("capture state leaked after Capture returned")
	}
}

func TestConcurrentCapturesShareThePoolSafely(t *testing.T) {
	if !Available() {
		t.Skip("cuda unavailable:", loadErrorText())
	}
	// Compile once outside capture: a capture must not perform first-time
	// compilation, so the shared kernel is prepared by the test goroutine.
	kernel, err := LoadKernel(graphKernelSource, "add_arrays")
	if err != nil {
		t.Fatal(err)
	}
	defer kernel.Close()
	entered := make(chan struct{}, 2)
	hold := make(chan struct{})
	launch := func(a, b, out *Buffer, n int) error {
		aAddr, bAddr, outAddr, size := a.Pointer(), b.Pointer(), out.Pointer(), int32(n)
		return kernel.Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, nil,
			[]unsafe.Pointer{unsafe.Pointer(&aAddr), unsafe.Pointer(&bAddr), unsafe.Pointer(&outAddr), unsafe.Pointer(&size)})
	}
	worker := func() error {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		fail := func(err error) error {
			entered <- struct{}{}
			<-hold
			return err
		}
		if err := SetDevice(0); err != nil {
			return fail(err)
		}
		const n = 256
		buffers := make([]*Buffer, 3)
		for i := range buffers {
			buffer, err := Alloc(n * 4)
			if err != nil {
				return fail(err)
			}
			defer buffer.Free()
			buffers[i] = buffer
		}
		a, b, out := buffers[0], buffers[1], buffers[2]
		stream, err := NewStream()
		if err != nil {
			return fail(err)
		}
		defer stream.Destroy()
		graph, err := Capture(stream, func() error {
			if err := launch(a, b, out, n); err != nil {
				return err
			}
			// Allocate and free scratch during capture so this thread's touched
			// bookkeeping is used while another thread captures at the same time.
			scratch, err := Alloc(64)
			if err != nil {
				return err
			}
			if err := launch(scratch, b, scratch, 16); err != nil {
				scratch.Free()
				return err
			}
			scratch.Free()
			entered <- struct{}{}
			<-hold
			return nil
		})
		if err != nil {
			return fail(err)
		}
		defer graph.Close()
		if err := graph.Launch(); err != nil {
			return err
		}
		return stream.Synchronize()
	}
	results := make(chan error, 2)
	go func() { results <- worker() }()
	go func() { results <- worker() }()
	// Both captures are live at the same time before either ends.
	<-entered
	<-entered
	close(hold)
	for range 2 {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	// The pool still hands out correct, device-matched blocks afterwards.
	buffer, err := Alloc(128)
	if err != nil {
		t.Fatal(err)
	}
	if buffer.Device() != 0 {
		t.Fatalf("post-capture buffer device %d", buffer.Device())
	}
	if err := buffer.Free(); err != nil {
		t.Fatal(err)
	}
}
