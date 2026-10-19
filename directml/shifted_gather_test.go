package directml

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/yh2237/gograd/gpu"
)

type shiftedGatherDriver struct {
	*testDriver
	copies       []gpu.View
	identity     int
	identityFrom resourceBinding
}

func (d *shiftedGatherDriver) copyView(out, src *allocation, v gpu.View, scatter bool) error {
	d.copies = append(d.copies, v)
	return d.testDriver.copyView(out, src, v, scatter)
}

func (d *shiftedGatherDriver) executeAt(key opKey, out *allocation, in ...resourceBinding) error {
	if key.kind == opIdentity {
		d.identity++
		d.identityFrom = in[0]
	}
	return d.testDriver.executeAt(key, out, in...)
}

func TestDirectMLShiftedGatherPlan(t *testing.T) {
	for _, tc := range []struct {
		name   string
		shape  []int
		stride []int
		length int
		staged bool
	}{
		{"transpose", []int{16, 16}, []int{1, 16}, 257, true},
		{"broadcast", []int{16, 16}, []int{0, 1}, 17, true},
		{"small", []int{4, 4}, []int{1, 4}, 17, false},
		{"sparse", []int{16, 16}, []int{1, 1024}, 15377, false},
		{"compact", []int{16, 16}, []int{16, 1}, 257, false},
		{"rank5", []int{1, 1, 1, 16, 16}, []int{256, 256, 256, 1, 16}, 257, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, base := fixture()
			d := &shiftedGatherDriver{testDriver: base}
			b.driver = d
			t.Cleanup(func() {
				if err := b.Close(); err != nil {
					t.Error(err)
				}
			})
			src := alloc(t, b, tc.length)
			n, _ := gpu.Elements(tc.shape...)
			out := alloc(t, b, n)
			before := b.MemoryStats()
			v := gpu.View{Buffer: src, Shape: tc.shape, Strides: tc.stride, Offset: 1}
			if err := b.Gather(out, v); err != nil {
				t.Fatal(err)
			}
			if len(d.copies) != 1 {
				t.Fatalf("copy calls=%d", len(d.copies))
			}
			if tc.staged {
				if d.identity != 1 || len(d.copies[0].Shape) != 1 || d.identityFrom.offset != 0 {
					t.Fatalf("staging plan: copies=%v identity=%d offset=%d", d.copies, d.identity, d.identityFrom.offset)
				}
				if d.copies[0].Offset != 1 || d.identityFrom.memory == nil {
					t.Fatal("lost source offset or temporary allocation")
				}
			} else if d.identity != 0 {
				t.Fatal("unexpected staging for fallback view")
			}
			after := b.MemoryStats()
			if before.LiveBytes != after.LiveBytes || before.LiveBuffers != after.LiveBuffers {
				t.Fatalf("scratch leaked: before=%+v after=%+v", before, after)
			}
		})
	}
	if shiftedGatherSpan(gpu.View{Shape: []int{4097, 4097}, Strides: []int{1, 4097}}, 4097*4097) != 0 {
		t.Fatal("staging exceeded 64 MiB bound")
	}
}

func TestDirectMLShiftedGatherFailureCleanup(t *testing.T) {
	for _, step := range []int{1, 2} {
		t.Run(fmt.Sprint(step), func(t *testing.T) {
			b, d := fixture()
			src, err := b.Alloc(257)
			if err != nil {
				t.Fatal(err)
			}
			out, err := b.Alloc(256)
			if err != nil {
				t.Fatal(err)
			}
			d.failure, d.failAt = errors.New("shifted gather failure"), step
			err = b.Gather(out, gpu.View{Buffer: src, Shape: []int{16, 16}, Strides: []int{1, 16}, Offset: 1})
			if !errors.Is(err, d.failure) || b.Available() {
				t.Fatalf("err=%v available=%t", err, b.Available())
			}
			if b.MemoryStats().LiveBuffers != 2 {
				t.Fatal("scratch was not released")
			}
			src.Close()
			out.Close()
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if d.allocated != d.freed {
				t.Fatalf("allocated=%d freed=%d", d.allocated, d.freed)
			}
		})
	}
}

func TestDirectMLShiftedGatherBits(t *testing.T) {
	b := testBackend(t)
	const rows, cols = 17, 19
	bits := []uint32{0x80000000, 0x7fc01234, 0x7fc54321, 0xff800000, 0x3f800000, 0}
	values := make([]float32, rows*cols+3)
	for i := range values {
		values[i] = math.Float32frombits(bits[i%len(bits)])
	}
	src := upload(t, b, values)
	out := alloc(t, b, rows*cols)
	before := b.MemoryStats()
	for offset := 1; offset <= 3; offset++ {
		v := gpu.View{Buffer: src, Shape: []int{rows, cols}, Strides: []int{1, rows}, Offset: offset}
		if err := b.Gather(out, v); err != nil {
			t.Fatal(err)
		}
		got, err := b.Download(out, rows*cols)
		if err != nil {
			t.Fatal(err)
		}
		for i, value := range got {
			want := values[viewIndex(i, v)]
			if math.Float32bits(value) != math.Float32bits(want) {
				t.Fatalf("offset=%d [%d] bits=%08x want=%08x", offset, i, math.Float32bits(value), math.Float32bits(want))
			}
		}
		if after := b.MemoryStats(); after.LiveBuffers != before.LiveBuffers || after.LiveBytes != before.LiveBytes {
			t.Fatal("staged gather leaked GPU memory")
		}
	}
}

func BenchmarkDirectMLGatherOffset(b *testing.B) {
	for _, offset := range []int{0, 1} {
		b.Run(fmt.Sprint(offset), func(b *testing.B) {
			backend := NewBackend()
			defer backend.Close()
			if !backend.Available() {
				b.Skip(backend.Err())
			}
			src, err := backend.Upload(make([]float32, 256*256+1))
			if err != nil {
				b.Fatal(err)
			}
			defer src.Close()
			out, err := backend.Alloc(256 * 256)
			if err != nil {
				b.Fatal(err)
			}
			defer out.Close()
			v := gpu.View{Buffer: src, Shape: []int{256, 256}, Strides: []int{1, 256}, Offset: offset}
			if err := backend.Gather(out, v); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := backend.Gather(out, v); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
