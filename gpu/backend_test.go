package gpu

import (
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
)

// The tests below exercise registry/ownership only, never operator execution.
type testBackend struct {
	Backend
	name      string
	available bool
}

func (b *testBackend) Name() string    { return b.name }
func (b *testBackend) Available() bool { return b.available }

type nonComparableBackend struct {
	Backend
	data []int
}

func (nonComparableBackend) Name() string { return "invalid" }

func TestRegistry(t *testing.T) {
	var r Registry
	alpha, beta := &testBackend{name: "alpha", available: true}, &testBackend{name: "beta"}
	if err := r.Register(beta); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(alpha); err != nil {
		t.Fatal(err)
	}
	if b, err := r.Get("alpha"); err != nil || b != alpha {
		t.Fatalf("Get: %v, %v", b, err)
	}
	for _, name := range []string{"beta", "missing", "cpu"} {
		if _, err := r.Get(name); !errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	if names := r.Names(); !reflect.DeepEqual(names, []string{"alpha", "beta"}) {
		t.Fatal(names)
	}
	names := r.Names()
	names[0] = "changed"
	if r.Names()[0] != "alpha" {
		t.Fatal("names aliases registry")
	}
	if err := r.Register(&testBackend{name: "alpha"}); err == nil {
		t.Fatal("replaced live registration")
	}
	if b, _ := r.Lookup("alpha"); b != alpha {
		t.Fatal("owner changed")
	}
	var nilBackend *testBackend
	for _, invalid := range []Backend{nil, nilBackend, nonComparableBackend{}, &testBackend{}, &testBackend{name: "cpu"}, &testBackend{name: "CUDA"}, &testBackend{name: "bad/name"}} {
		if err := r.Register(invalid); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("invalid %T: %v", invalid, err)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if b, ok := r.Lookup("alpha"); !ok || b != alpha {
					t.Error("concurrent lookup")
				}
				if err := r.Register(alpha); err == nil {
					t.Error("concurrent duplicate registered")
				}
			}
		}()
	}
	wg.Wait()
}

func TestBufferOwnershipAndClose(t *testing.T) {
	a, b := &testBackend{name: "same"}, &testBackend{name: "same"}
	var freed atomic.Int32
	native := []float32{1, 2, 3}
	buffer, err := NewBuffer(a, len(native), native, func() { freed.Add(1) })
	if err != nil {
		t.Fatal(err)
	}
	if buffer.Backend() != a || buffer.Len() != 3 {
		t.Fatal("buffer metadata")
	}
	if err := CheckBuffer(a, buffer, 3); err != nil {
		t.Fatal(err)
	}
	if err := CheckBuffer(a, buffer, 2); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := buffer.Native(b); !errors.Is(err, ErrBackendMismatch) {
		t.Fatalf("same name was mistaken for same owner: %v", err)
	}
	if value, err := buffer.Native(a); err != nil || !reflect.DeepEqual(value, native) {
		t.Fatalf("native: %v %v", value, err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); buffer.Close() }()
	}
	wg.Wait()
	if freed.Load() != 1 {
		t.Fatalf("freed %d times", freed.Load())
	}
	if _, err := buffer.Native(a); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := CheckBuffer(a, nil, 0); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	var absent *Buffer
	absent.Close()
	if _, err := NewBuffer(a, -1, nil, func() {}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := NewBuffer(a, 0, nil, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := NewBuffer(nil, 0, nil, func() {}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}
}

func TestElements(t *testing.T) {
	for _, tc := range []struct {
		shape []int
		want  int
	}{
		{nil, 1}, {[]int{2, 3, 4}, 24}, {[]int{0, 5}, 0}, {[]int{maxElements}, maxElements},
	} {
		if n, err := Elements(tc.shape...); err != nil || n != tc.want {
			t.Fatalf("%v: %d %v", tc.shape, n, err)
		}
	}
	for _, shape := range [][]int{{-1}, {0, -1}, {maxElements, 2}, {maxElements + 1}} {
		if _, err := Elements(shape...); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("%v: %v", shape, err)
		}
	}
}

func TestConv1dGeometry(t *testing.T) {
	o := Conv1dOptions{Batch: 2, InputLength: 11, InChannels: 3, OutChannels: 5, Kernel: 4, Stride: 2, Padding: 3, Dilation: 2}
	if n, err := o.OutputLength(); err != nil || n != 6 {
		t.Fatalf("%d %v", n, err)
	}
	for _, change := range []func(*Conv1dOptions){
		func(o *Conv1dOptions) { o.Stride = 0 },
		func(o *Conv1dOptions) { o.Dilation = 0 },
		func(o *Conv1dOptions) { o.Padding = -1 },
		func(o *Conv1dOptions) { o.Kernel = 100 },
		func(o *Conv1dOptions) { o.Batch = 0 },
		func(o *Conv1dOptions) { o.Padding = maxElements },
	} {
		invalid := o
		change(&invalid)
		if _, err := invalid.OutputLength(); !errors.Is(err, ErrInvalidArgument) {
			t.Fatal(err)
		}
	}
	// Validate each allocation separately, not the unrelated product of every
	// convolution dimension (which would reject normal large convolutions).
	o = Conv1dOptions{Batch: 16, InputLength: 768, InChannels: 768, OutChannels: 3072, Kernel: 3, Stride: 1, Padding: 1, Dilation: 1}
	if n, err := o.OutputLength(); err != nil || n != 768 {
		t.Fatalf("%d %v", n, err)
	}
}
