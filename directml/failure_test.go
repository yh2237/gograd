package directml

import (
	"errors"
	"fmt"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

// These tests do not need Windows, COM, or a GPU. The fake checks ownership and
// error cleanup only; production operations never evaluate tensors on the CPU.
type testDriver struct {
	failure                                     error
	allocated, freed, closed, releasedOperators int
	computeCalls, failAt                        int
	lastKey                                     opKey
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
func (d *testDriver) execute(key opKey, out *allocation, inputs ...*allocation) error {
	d.lastKey = key
	return d.computeFailure()
}
func (d *testDriver) copyView(out, src *allocation, v gpu.View, scatter bool) error {
	return d.computeFailure()
}
func (d *testDriver) computeFailure() error {
	d.computeCalls++
	if d.failAt == 0 || d.computeCalls == d.failAt {
		return d.failure
	}
	return nil
}
func (*testDriver) operatorStats() OperatorStats { return OperatorStats{} }
func (d *testDriver) releaseOperators()          { d.releasedOperators++ }
func (d *testDriver) close()                     { d.closed++ }
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

func TestDirectMLPortableM5Failures(t *testing.T) {
	for _, tc := range []struct {
		name string
		at   int
	}{{"view_gemm", 1}, {"view_gemm", 2}, {"view_gemm", 3}, {"gelu", 1}, {"gelu", 6}, {"gelu", 9}, {"norm", 1}, {"norm", 2}, {"norm", 3}} {
		t.Run(fmt.Sprintf("%s/%d", tc.name, tc.at), func(t *testing.T) {
			b, d := fixture()
			inputSize := 4
			if tc.name == "view_gemm" {
				inputSize = 5
			}
			x, err := b.Alloc(inputSize)
			if err != nil {
				t.Fatal(err)
			}
			w, err := b.Alloc(inputSize)
			if err != nil {
				t.Fatal(err)
			}
			y, err := b.Alloc(4)
			if err != nil {
				t.Fatal(err)
			}
			d.failure = errors.New("injected M5 step failure")
			d.failAt = tc.at
			switch tc.name {
			case "view_gemm":
				a := gpu.View{Buffer: x, Shape: []int{2, 2}, Strides: []int{1, 2}, Offset: 1}
				right := a
				right.Buffer = w
				err = b.MatMulViews(y, a, right)
			case "gelu":
				err = b.GELU(y, x, 4)
			case "norm":
				err = b.LayerNorm(y, x, w, w, 1, 4, 1e-5)
			}
			if !errors.Is(err, d.failure) || b.Available() {
				t.Fatalf("%v, available=%t", err, b.Available())
			}
			if stats := b.MemoryStats(); stats.LiveBuffers != 3 || stats.CachedBuffers != 0 {
				t.Fatalf("scratch survived failure: %+v", stats)
			}
			x.Close()
			w.Close()
			y.Close()
			if d.allocated != d.freed {
				t.Fatalf("%+v", d)
			}
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDirectMLPortableFeatureGate(t *testing.T) {
	b, d := fixture()
	x, err := b.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	y, err := b.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.GELUExact(y, x, 4); !errors.Is(err, gpu.ErrUnsupported) {
		t.Fatal(err)
	}
	if !b.Available() || d.computeCalls != 0 {
		t.Fatal("unsupported feature was dispatched or disabled the device")
	}
	x.Close()
	y.Close()
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestDirectMLPortableStridedGEMM(t *testing.T) {
	b, d := fixture()
	x, err := b.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	y, err := b.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	v := gpu.View{Buffer: x, Shape: []int{2, 2}, Strides: []int{1, 2}}
	if err := b.MatMulViews(y, v, v); err != nil {
		t.Fatal(err)
	}
	if d.allocated != 2 || d.computeCalls != 1 || d.lastKey.aLayout.strides[2] != 1 || d.lastKey.aLayout.strides[3] != 2 {
		t.Fatal("strided GEMM allocated copies instead of describing the input", d.lastKey)
	}
	x.Close()
	y.Close()
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}
