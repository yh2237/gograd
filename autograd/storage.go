package autograd

import (
	"errors"
	"sync/atomic"

	"github.com/yh2237/gograd/cuda"
)

// tensorStorage owns one allocation. Each tensor/view has a lease; returning
// an allocation to a pool is deferred until its last lease is released.
type tensorStorage struct {
	data       []float32
	buffer     *cuda.Buffer
	pooledCPU  bool
	references atomic.Int64
	version    atomic.Uint64
}

func cpuStorage(data []float32, pooled bool) *tensorStorage {
	s := &tensorStorage{data: data, pooledCPU: pooled}
	s.references.Store(1)
	return s
}

func deviceStorage(buffer *cuda.Buffer) *tensorStorage {
	s := &tensorStorage{buffer: buffer}
	s.references.Store(1)
	return s
}

func (s *tensorStorage) acquire() *tensorStorage {
	if s.references.Add(1) <= 1 {
		panic("autograd: acquiring released storage")
	}
	return s
}

func (s *tensorStorage) release() {
	if s == nil {
		return
	}
	refs := s.references.Add(-1)
	if refs < 0 {
		panic("autograd: storage released twice")
	}
	if refs != 0 {
		return
	}
	if s.pooledCPU {
		cpuRelease(s.data)
	}
	if s.buffer != nil {
		s.buffer.Free()
	}
	s.data, s.buffer = nil, nil
}

func (t *Tensor) checkOpen() error {
	if t == nil || t.closed.Load() || t.disposed.Load() {
		return errors.New("autograd: tensor is closed")
	}
	return nil
}

func (t *Tensor) assertOpen() {
	if err := t.checkOpen(); err != nil {
		panic(err)
	}
}

// Parent edges preserve both the parent object and its allocation: backward
// closures read those objects even if the caller closes its original handle.
func (t *Tensor) attachParents(parents []*Tensor) {
	t.parents = append([]*Tensor(nil), parents...)
	t.gradParents = t.parents
	t.savedVersions = make([]uint64, len(parents))
	for i, p := range parents {
		p.assertOpen()
		p.children.Add(1)
		t.savedVersions[i] = p.storage.version.Load()
	}
}

// keepAlive adds a non-differentiable dependency, e.g. a loss target read by
// backward. Its lifetime matters even though gradients do not flow into it.
func (t *Tensor) keepAlive(parent *Tensor) {
	parent.assertOpen()
	t.parents = append(t.parents, parent)
	t.savedVersions = append(t.savedVersions, parent.storage.version.Load())
	parent.children.Add(1)
}

func (t *Tensor) releaseSaved() {
	for _, b := range t.aux {
		b.Free()
	}
	t.aux = nil
	for _, data := range t.auxCPU {
		cpuRelease(data)
	}
	t.auxCPU = nil
	for _, lease := range t.savedLeases {
		lease.release()
	}
	t.savedLeases = nil
	t.backward, t.backwardGPU = nil, nil
	if t.intermediate {
		t.historyReleased = true
	}
}

func (t *Tensor) dropParents() {
	parents := t.parents
	t.parents, t.gradParents, t.savedVersions = nil, nil, nil
	for _, p := range parents {
		if p.children.Add(-1) < 0 {
			panic("autograd: parent released twice")
		}
		p.maybeDispose()
	}
}

// maybeDispose runs after the last dependent is done. Data retention keeps the
// allocation but does not retain an explicitly released derivative history.
func (t *Tensor) maybeDispose() {
	if t.children.Load() != 0 || t.inBackward.Load() || t.disposed.Load() {
		return
	}
	if !t.closed.Load() && !t.releaseRequested.Load() {
		return
	}
	if !t.closed.Load() && (t.retain || t.retainData) {
		t.releaseSaved()
		t.dropParents()
		return
	}
	if !t.disposed.CompareAndSwap(false, true) {
		return
	}
	t.closed.Store(true)
	t.releaseSaved()
	t.dropBF16()
	if t.gradBuf != nil {
		t.gradBuf.Free()
		t.gradBuf = nil
	}
	t.Grad = nil
	t.storage.release()
	t.storage, t.Data, t.buf = nil, nil, nil
	t.dropParents()
}

// RetainData keeps a tensor's forward values after graph release/backward.
// It does not retain history; Close releases the retained handle explicitly.
func (t *Tensor) RetainData() {
	t.assertOpen()
	t.retainData = true
}

// collectLifetime follows every dependency, including non-gradient inputs.
func (t *Tensor) collectLifetime() []*Tensor {
	seen := make(map[*Tensor]bool)
	var out []*Tensor
	var visit func(*Tensor)
	visit = func(v *Tensor) {
		if seen[v] || v.disposed.Load() {
			return
		}
		seen[v] = true
		out = append(out, v)
		for _, p := range v.parents {
			visit(p)
		}
	}
	visit(t)
	return out
}
