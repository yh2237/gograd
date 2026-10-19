//go:build (windows || (linux && !android)) && amd64

package vulkan

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"unsafe"

	"github.com/yh2237/gograd/gpu"
)

// This harness exercises the actual native submission/allocation/cleanup code,
// replacing only the Vulkan ABI boundary. It never opens a loader or a GPU.
type nativeRecorder struct {
	results                   map[string][]int32
	calls                     map[string]int
	next                      uint64
	sizes                     map[uint64]uint64
	memory                    map[uint64][]float32
	descriptors               map[uint64][4]vkDescriptorBufferInfo
	sets                      []uint64
	pushes                    [][]byte
	barriers                  []nativeBarrier
	dispatches                int
	poolSets, poolDescriptors uint32
}

type nativeBarrier struct {
	vkBufferBarrier
	afterDispatch      int
	srcStage, dstStage uint32
}

func newNativeRecorder() (*vkDriver, *nativeRecorder) {
	r := &nativeRecorder{
		results: make(map[string][]int32), calls: make(map[string]int), next: 100,
		sizes: make(map[uint64]uint64), memory: make(map[uint64][]float32),
		descriptors: make(map[uint64][4]vkDescriptorBufferInfo),
	}
	d := &vkDriver{
		device: 1, instance: 2, queue: 3, command: 4, commandPool: 5, fence: 6,
		descriptorPool: 7, descriptorLayout: 8, descriptorSet: 10,
		descriptorSets: [3]uint64{10, 11, 12}, pipelineLayout: 13,
		pipeline: 14, convPipeline: 15, gemmPipelines: [4]uint64{16, 17, 18, 19},
		properties: Info{MaxStorageBufferBytes: 1 << 30, MaxWorkGroupCount: [3]uint32{65535, 65535, 65535}},
		procs:      map[string]uintptr{"vkDestroyInstance": 1}, callHook: r.call,
	}
	d.memory.TypeCount = 1
	d.memory.Types[0].Flags = 1 | 2 | 4
	return d, r
}

// Like the real Vulkan ABI, the test boundary receives live pointer bit patterns.
// Reinterpret argument storage, without pointer arithmetic. vkDriver.call's
// uintptrescapes directive keeps the pointed-to allocations live through the call.
func nativeArgPointer(arg uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&arg))
}

func (r *nativeRecorder) call(name string, a ...uintptr) uintptr {
	r.calls[name]++
	if codes := r.results[name]; len(codes) != 0 {
		r.results[name] = codes[1:]
		if codes[0] != 0 {
			return uintptr(int64(codes[0]))
		}
	}
	switch name {
	case "vkResetCommandBuffer":
		r.dispatches = 0
	case "vkCreateDescriptorSetLayout", "vkCreateDescriptorPool", "vkCreatePipelineLayout", "vkCreateShaderModule":
		r.next++
		*(*uint64)(nativeArgPointer(a[3])) = r.next
		if name == "vkCreateDescriptorPool" {
			pool := (*vkDescriptorPoolCreateInfo)(nativeArgPointer(a[1]))
			r.poolSets, r.poolDescriptors = pool.MaxSets, pool.Sizes.Count
		}
	case "vkAllocateDescriptorSets":
		info := (*vkDescriptorAllocateInfo)(nativeArgPointer(a[1]))
		for i := range unsafe.Slice((*uint64)(nativeArgPointer(a[2])), int(info.Count)) {
			r.next++
			unsafe.Slice((*uint64)(nativeArgPointer(a[2])), int(info.Count))[i] = r.next
		}
	case "vkCreateComputePipelines":
		r.next++
		*(*uint64)(nativeArgPointer(a[5])) = r.next
	case "vkCreateBuffer":
		r.next++
		*(*uint64)(nativeArgPointer(a[3])) = r.next
		r.sizes[r.next] = (*vkBufferCreateInfo)(nativeArgPointer(a[1])).Size
	case "vkGetBufferMemoryRequirements":
		*(*vkMemoryRequirements)(nativeArgPointer(a[2])) = vkMemoryRequirements{Size: r.sizes[uint64(a[1])], TypeBits: 1}
	case "vkAllocateMemory":
		r.next++
		*(*uint64)(nativeArgPointer(a[3])) = r.next
		size := (*vkMemoryAllocateInfo)(nativeArgPointer(a[1])).Size
		r.memory[r.next] = make([]float32, size/4)
	case "vkMapMemory":
		*(*unsafe.Pointer)(nativeArgPointer(a[5])) = unsafe.Pointer(&r.memory[uint64(a[1])][0])
	case "vkFreeMemory":
		delete(r.memory, uint64(a[1]))
	case "vkDestroyBuffer":
		delete(r.sizes, uint64(a[1]))
	case "vkUpdateDescriptorSets":
		for _, w := range unsafe.Slice((*vkDescriptorWrite)(nativeArgPointer(a[2])), int(a[1])) {
			bindings := r.descriptors[w.Set]
			bindings[w.Binding] = *w.BufferInfo
			r.descriptors[w.Set] = bindings
		}
	case "vkCmdBindDescriptorSets":
		r.sets = append(r.sets, *(*uint64)(nativeArgPointer(a[5])))
	case "vkCmdPushConstants":
		r.pushes = append(r.pushes, append([]byte(nil), unsafe.Slice((*byte)(nativeArgPointer(a[5])), int(a[4]))...))
	case "vkCmdPipelineBarrier":
		r.barriers = append(r.barriers, nativeBarrier{*(*vkBufferBarrier)(nativeArgPointer(a[7])), r.dispatches, uint32(a[1]), uint32(a[2])})
	case "vkCmdDispatch":
		r.dispatches++
	}
	return 0
}

func recorderBackend(d *vkDriver) *Backend {
	b := NewBackend()
	b.once.Do(func() { b.driver = d })
	return b
}

func recorderAlloc(t *testing.T, b *Backend, n int) *gpu.Buffer {
	t.Helper()
	x, err := b.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func TestNativeConv1dDescriptorSlots(t *testing.T) {
	d, r := newNativeRecorder()
	if err := d.initPipeline(); err != nil {
		t.Fatal(err)
	}
	if r.poolSets != 3 || r.poolDescriptors != 12 || d.descriptorSet != d.descriptorSets[0] {
		t.Fatalf("pool=%d/%d sets=%v default=%d", r.poolSets, r.poolDescriptors, d.descriptorSets, d.descriptorSet)
	}
	seen := make(map[uint64]bool)
	for _, set := range d.descriptorSets {
		if set == 0 || seen[set] {
			t.Fatalf("non-independent descriptor slots: %v", d.descriptorSets)
		}
		seen[set] = true
	}
	if err := d.close(); err != nil {
		t.Fatal(err)
	}
}

func TestNativeSubmissionFailures(t *testing.T) {
	for _, tc := range []struct {
		name, op   string
		code, idle int32
		pending    bool
		waits      int
	}{
		{"reset", "vkResetCommandBuffer", -1, 0, false, 0},
		{"begin", "vkBeginCommandBuffer", -1, 0, false, 0},
		{"end", "vkEndCommandBuffer", -1, 0, false, 0},
		{"resetFence", "vkResetFences", -1, 0, false, 0},
		{"queueDrained", "vkQueueSubmit", -1, 0, false, 1},
		{"waitDrained", "vkWaitForFences", -1, 0, false, 1},
		{"queueLost", "vkQueueSubmit", -4, 0, false, 0},
		{"waitLost", "vkWaitForFences", -4, 0, false, 0},
		{"emergencyLost", "vkWaitForFences", -1, -4, false, 1},
		{"queueUnknown", "vkQueueSubmit", -1, -2, true, 1},
		{"waitUnknown", "vkWaitForFences", -1, -2, true, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, r := newNativeRecorder()
			r.results[tc.op] = []int32{tc.code}
			r.results["vkDeviceWaitIdle"] = []int32{tc.idle}
			err := d.submit(func() {})
			if err == nil || !strings.Contains(err.Error(), tc.op) || d.pending != tc.pending {
				t.Fatalf("err=%v pending=%t", err, d.pending)
			}
			if r.calls["vkDeviceWaitIdle"] != tc.waits {
				t.Fatalf("idle waits=%d, want %d", r.calls["vkDeviceWaitIdle"], tc.waits)
			}
			if tc.pending && !strings.Contains(err.Error(), "emergency vkDeviceWaitIdle") {
				t.Fatal("emergency failure was discarded:", err)
			}
			calls := r.calls["vkResetCommandBuffer"]
			if err := d.submit(func() { t.Error("recorded on failed device") }); err == nil || r.calls["vkResetCommandBuffer"] != calls {
				t.Fatal("failed native driver reused command state")
			}
			memory := &block{buffer: 30, memory: 31}
			if freed := d.free(memory); freed == tc.pending {
				t.Fatalf("freed=%t pending=%t", freed, tc.pending)
			}
			if err := d.close(); err != nil {
				t.Fatal(err)
			}
			if memory.buffer != 0 || r.calls["vkDestroyDevice"] != 1 {
				t.Fatal("safe cleanup did not reclaim resources")
			}
		})
	}
}

func TestNativeQuarantineSurvivesPoolAndClose(t *testing.T) {
	for _, finalIdle := range []int32{0, -4} {
		t.Run(fmt.Sprint(finalIdle), func(t *testing.T) {
			d, r := newNativeRecorder()
			b := recorderBackend(d)
			cached := recorderAlloc(t, b, 129)
			cached.Close()
			x := recorderAlloc(t, b, 3)
			r.results["vkWaitForFences"] = []int32{-1}
			r.results["vkDeviceWaitIdle"] = []int32{-2, -2, finalIdle}
			err := b.CopyFrom(x, []float32{1, 2, 3})
			if err == nil || !strings.Contains(err.Error(), "vkWaitForFences") || !strings.Contains(err.Error(), "emergency vkDeviceWaitIdle") {
				t.Fatal(err)
			}
			if b.Available() || !errors.Is(b.Err(), gpu.ErrUnavailable) {
				t.Fatal("backend still available")
			}
			if err := b.Zero(x); !errors.Is(err, gpu.ErrUnavailable) {
				t.Fatal(err)
			}
			x.Close()
			b.ReleasePool()
			b.SetPoolCacheLimit(0)
			s := b.MemoryStats()
			if s.LiveBuffers != 0 || s.CachedBuffers != 0 || s.QuarantinedBuffers != 3 || s.QuarantinedBytes != s.ReservedBytes || s.ReservedBytes == 0 {
				t.Fatalf("quarantine accounting: %+v", s)
			}
			if err := b.Close(); !errors.Is(err, gpu.ErrUnavailable) {
				t.Fatal("Close must report unproven completion:", err)
			}
			for _, name := range []string{"vkUnmapMemory", "vkDestroyBuffer", "vkFreeMemory", "vkDestroyDevice", "vkDestroyInstance", "vkDestroyCommandPool", "vkDestroyDescriptorPool", "vkDestroyFence", "vkDestroyPipeline"} {
				if r.calls[name] != 0 {
					t.Fatalf("unsafe %s called %d times", name, r.calls[name])
				}
			}
			if d.device == 0 || d.instance == 0 || len(d.quarantine) != 3 || r.calls["vkDeviceWaitIdle"] != 2 {
				t.Fatal("Close discarded quarantine or retried without bound")
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if r.calls["vkDeviceWaitIdle"] != 3 || r.calls["vkDestroyBuffer"] != 3 || r.calls["vkFreeMemory"] != 3 || len(r.memory) != 0 {
				t.Fatalf("cleanup calls: %v, memory=%d", r.calls, len(r.memory))
			}
			if s := b.MemoryStats(); s.ReservedBytes != 0 || s.QuarantinedBuffers != 0 || s.QuarantinedBytes != 0 {
				t.Fatalf("cleanup accounting: %+v", s)
			}
			if err := b.Close(); err != nil || r.calls["vkDestroyDevice"] != 1 {
				t.Fatal("Close not idempotent:", err)
			}
		})
	}
}

func TestNativeConv1dBatch(t *testing.T) {
	for _, bias := range []bool{false, true} {
		t.Run(fmt.Sprint(bias), func(t *testing.T) {
			d, r := newNativeRecorder()
			b := recorderBackend(d)
			x, w, out := recorderAlloc(t, b, 18), recorderAlloc(t, b, 12), recorderAlloc(t, b, 14)
			var v *gpu.Buffer
			if bias {
				v = recorderAlloc(t, b, 2)
			}
			o := gpu.Conv1dOptions{Batch: 1, InputLength: 6, InChannels: 3, OutChannels: 2, Kernel: 2, Stride: 1, Padding: 1, Dilation: 1}
			stages := 2
			if bias {
				stages++
			}
			for iteration := 0; iteration < 3; iteration++ {
				r.sets, r.pushes, r.barriers = nil, nil, nil
				before := r.calls["vkQueueSubmit"]
				allocated := r.calls["vkAllocateMemory"]
				if err := b.Conv1d(out, x, w, v, o); err != nil {
					t.Fatal(err)
				}
				if r.calls["vkQueueSubmit"]-before != 1 || r.calls["vkWaitForFences"] != iteration+1 || len(r.sets) != stages || len(r.pushes) != stages {
					t.Fatalf("submits=%d sets=%v pushes=%d", r.calls["vkQueueSubmit"]-before, r.sets, len(r.pushes))
				}
				if iteration > 0 && r.calls["vkAllocateMemory"] != allocated {
					t.Fatal("warm batch allocated memory")
				}
				for i, set := range r.sets {
					if set != d.descriptorSets[i] {
						t.Fatalf("descriptor slot %d: %d", i, set)
					}
				}
				im2col, gemm := r.descriptors[r.sets[0]], r.descriptors[r.sets[1]]
				if im2col[2].Buffer != gemm[0].Buffer || im2col[0].Buffer == gemm[0].Buffer || gemm[1].Buffer == im2col[1].Buffer {
					t.Fatalf("overwritten descriptors: im2col=%v gemm=%v", im2col, gemm)
				}
				cp := *(*convParams)(unsafe.Pointer(&r.pushes[0][0]))
				gp := *(*gemmParams)(unsafe.Pointer(&r.pushes[1][0]))
				if cp.Kind != 0 || cp.N != 42 || gp.M != 7 || gp.N != 2 || gp.K != 6 {
					t.Fatalf("push constants: %+v %+v", cp, gp)
				}
				if bias {
					bp := *(*convParams)(unsafe.Pointer(&r.pushes[2][0]))
					bindings := r.descriptors[r.sets[2]]
					if bp.Kind != 1 || bp.N != 14 || bindings[0].Buffer != gemm[2].Buffer || bindings[2].Buffer != gemm[2].Buffer {
						t.Fatalf("bias constants/bindings: %+v %v", bp, bindings)
					}
				}
				colBarrier, biasBarrier := false, !bias
				for _, barrier := range r.barriers {
					if barrier.SrcAccess == accessShaderWrite && barrier.DstAccess&accessShaderRead != 0 && barrier.srcStage == stageCompute && barrier.dstStage == stageCompute {
						colBarrier = colBarrier || (barrier.Buffer == im2col[2].Buffer && barrier.afterDispatch == 1)
						biasBarrier = biasBarrier || (barrier.Buffer == gemm[2].Buffer && barrier.DstAccess&accessShaderWrite != 0 && barrier.afterDispatch == 2)
					}
				}
				if !colBarrier || !biasBarrier {
					t.Fatal("missing inter-dispatch dependencies")
				}
			}
			x.Close()
			w.Close()
			out.Close()
			if v != nil {
				v.Close()
			}
			if err := b.Close(); err != nil || len(r.memory) != 0 {
				t.Fatalf("cleanup: %v allocations=%d", err, len(r.memory))
			}
		})
	}
}

func TestNativeConv1dFailures(t *testing.T) {
	for _, bias := range []bool{false, true} {
		for _, tc := range []struct {
			op      string
			idle    int32
			pending bool
		}{
			{"vkResetCommandBuffer", 0, false},
			{"vkBeginCommandBuffer", 0, false},
			{"vkEndCommandBuffer", 0, false},
			{"vkResetFences", 0, false},
			{"vkQueueSubmit", 0, false},
			{"vkWaitForFences", 0, false},
			{"vkQueueSubmit", -2, true},
			{"vkWaitForFences", -2, true},
		} {
			t.Run(fmt.Sprintf("bias=%t/%s/idle=%d", bias, tc.op, tc.idle), func(t *testing.T) {
				d, r := newNativeRecorder()
				b := recorderBackend(d)
				x, w, out := recorderAlloc(t, b, 18), recorderAlloc(t, b, 12), recorderAlloc(t, b, 14)
				var v *gpu.Buffer
				if bias {
					v = recorderAlloc(t, b, 2)
				}
				r.results[tc.op] = []int32{-1}
				r.results["vkDeviceWaitIdle"] = []int32{tc.idle}
				o := gpu.Conv1dOptions{Batch: 1, InputLength: 6, InChannels: 3, OutChannels: 2, Kernel: 2, Stride: 1, Padding: 1, Dilation: 1}
				err := b.Conv1d(out, x, w, v, o)
				if err == nil || !strings.Contains(err.Error(), tc.op) || b.Available() || d.pending != tc.pending {
					t.Fatalf("err=%v pending=%t", err, d.pending)
				}
				before := r.calls["vkUpdateDescriptorSets"]
				if err := d.conv1d(convBatch{}); err == nil || r.calls["vkUpdateDescriptorSets"] != before {
					t.Fatal("failed native batch reused descriptor state")
				}
				x.Close()
				w.Close()
				out.Close()
				if v != nil {
					v.Close()
				}
				b.ReleasePool()
				s := b.MemoryStats()
				if s.LiveBuffers != 0 || s.CachedBuffers != 0 {
					t.Fatalf("leaked scratch: %+v", s)
				}
				if tc.pending {
					if r.calls["vkFreeMemory"] != 0 || s.ReservedBytes == 0 || s.ReservedBytes != s.QuarantinedBytes {
						t.Fatalf("unsafe batch cleanup: %+v", s)
					}
				} else if s.ReservedBytes != 0 {
					t.Fatalf("batch resources leaked: %+v", s)
				}
				if err := b.Close(); err != nil || len(r.memory) != 0 || len(r.sizes) != 0 {
					t.Fatalf("cleanup: %v, memories=%d buffers=%d", err, len(r.memory), len(r.sizes))
				}
			})
		}
	}
}
