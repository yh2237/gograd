package directml

import (
	"errors"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

// These tests do not need Windows, COM, or a GPU. The fake checks ownership and
// error cleanup only; production operations never evaluate tensors on the CPU.
type testDriver struct {
	failure                                     error
	allocated, freed, closed, releasedOperators int
}

func (*testDriver) info() Info { return Info{Name: "fixture", FeatureLevel: 0x1000} }
func (d *testDriver) alloc(size uint64, heap uint32) (*allocation, error) {
	d.allocated++
	return &allocation{native: make([]float32, size/4), size: size, allocated: (size + 65535) &^ 65535, heap: heap}, nil
}
func (d *testDriver) free(*allocation) { d.freed++ }
func (d *testDriver) write(v *allocation, values []float32) error {
	if d.failure != nil {
		return d.failure
	}
	if values == nil {
		clear(v.native.([]float32))
	} else {
		copy(v.native.([]float32), values)
	}
	return nil
}
func (d *testDriver) read(v *allocation, values []float32) error {
	if d.failure != nil {
		return d.failure
	}
	copy(values, v.native.([]float32))
	return nil
}
func (d *testDriver) copy(out, src *allocation, n uint64) error {
	if d.failure != nil {
		return d.failure
	}
	copy(out.native.([]float32)[:n/4], src.native.([]float32)[:n/4])
	return nil
}
func (d *testDriver) execute(opKey, *allocation, *allocation, *allocation) error { return d.failure }
func (*testDriver) operatorStats() OperatorStats                                 { return OperatorStats{} }
func (d *testDriver) releaseOperators()                                          { d.releasedOperators++ }
func (d *testDriver) close()                                                     { d.closed++ }
func fixture() (*Backend, *testDriver) {
	d := &testDriver{}
	b := NewBackend()
	b.once.Do(func() { b.driver = d })
	return b, d
}

func TestDirectMLPortableLifetime(t *testing.T) {
	b, d := fixture()
	x, err := b.Upload([]float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if d.closed != 0 {
		t.Fatal("closed driver with live allocation")
	}
	values, err := b.Download(x, 2)
	if err != nil || len(values) != 2 || values[0] != 1 || values[1] != 2 {
		t.Fatalf("%v %v", values, err)
	}
	if _, err := b.Download(x, 4); !errors.Is(err, gpu.ErrInvalidArgument) {
		t.Fatal(err)
	}
	other, _ := fixture()
	if err := other.CopyFrom(x, []float32{1, 2, 3}); !errors.Is(err, gpu.ErrBackendMismatch) {
		t.Fatal(err)
	}
	other.Close()
	x.Close()
	x.Close()
	if stats := b.MemoryStats(); stats.LiveBuffers != 0 || stats.LiveBytes != 0 || stats.CachedBuffers != 3 {
		t.Fatal(stats)
	}
	allocations := d.allocated
	x, err = b.Alloc(3)
	if err != nil {
		t.Fatal(err)
	}
	if d.allocated != allocations {
		t.Fatal("did not reuse matching size class")
	}
	if err := b.Zero(x); err != nil {
		t.Fatal(err)
	}
	check(t, b, x, []float32{0, 0, 0})
	x.Close()
	b.SetPoolCacheLimit(0)
	if stats := b.MemoryStats(); stats.CachedBytes != 0 || stats.ReservedBytes != 0 {
		t.Fatal(stats)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if d.freed != d.allocated || d.closed != 1 {
		t.Fatalf("%+v", d)
	}
	if _, err := b.Alloc(1); !errors.Is(err, gpu.ErrClosed) {
		t.Fatal(err)
	}
}

func TestDirectMLPortableFailure(t *testing.T) {
	for _, op := range []string{"upload", "download", "execute"} {
		t.Run(op, func(t *testing.T) {
			b, d := fixture()
			x, err := b.Alloc(4)
			if err != nil {
				t.Fatal(err)
			}
			y, err := b.Alloc(4)
			if err != nil {
				t.Fatal(err)
			}
			d.failure = errors.New("injected DirectML failure")
			switch op {
			case "upload":
				err = b.CopyFrom(x, []float32{1, 2, 3, 4})
			case "download":
				_, err = b.Download(x, 4)
			case "execute":
				err = b.Add(y, x, x, 4)
			}
			if !errors.Is(err, d.failure) || b.Available() || !errors.Is(b.Err(), gpu.ErrUnavailable) {
				t.Fatalf("err=%v available=%t", err, b.Available())
			}
			x.Close()
			y.Close()
			if stats := b.MemoryStats(); stats.LiveBuffers != 0 || stats.ReservedBytes != 0 {
				t.Fatal(stats)
			}
			if d.allocated != d.freed || d.releasedOperators == 0 {
				t.Fatalf("%+v", d)
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDirectMLPortableCommittedLimit(t *testing.T) {
	b, d := fixture()
	b.SetPoolCacheLimit(256)
	x, err := b.Alloc(1)
	if err != nil {
		t.Fatal(err)
	}
	if s := b.MemoryStats(); s.LiveBytes != 65536 || s.ReservedBytes != 65536 {
		t.Fatal(s)
	}
	x.Close()
	if s := b.MemoryStats(); s.CachedBytes != 0 || s.ReservedBytes != 0 {
		t.Fatal("pool charged Width instead of committed allocation", s)
	}
	if d.freed != 1 {
		t.Fatal("oversized backing allocation was cached")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestDirectMLLazyRegistration(t *testing.T) {
	b, ok := gpu.Lookup("directml")
	if !ok || b != defaultBackend {
		t.Fatal("backend not registered")
	}
	if err := defaultBackend.Close(); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatal(err)
	}
	// Independent Close and statistics must not load a driver.
	independent := NewBackend()
	if stats := independent.MemoryStats(); stats.ReservedBytes != 0 {
		t.Fatal(stats)
	}
	independent.ReleaseOperators()
	independent.ReleasePool()
	if err := independent.Close(); err != nil {
		t.Fatal(err)
	}
}
