package torchpth

import (
	"strings"
	"testing"
)

func TestMalformedDecodeStack(t *testing.T) {
	for _, input := range [][]byte{
		{'r', 0, 0, 0, 0}, // LONG_BINPUT with no stack value
		{'q', 0}, {'.'}, {'R'}, {'Q'}, {'s'}, {'u'}, {'b'},
		{0x85}, {0x86, '.'}, {0x87}, {'t'},
		{'h', 0}, {'j', 0, 0, 0, 0},
		{'r', 0}, {'X', 10, 0, 0, 0}, {'c', 'x', '\n'},
	} {
		d := decoder{data: input, memo: map[int]any{}}
		if _, err := d.run(); err == nil {
			t.Fatalf("malformed %x accepted", input)
		}
	}
	d := decoder{data: []byte{'r', 0, 0, 0, 0}, memo: map[int]any{}}
	if _, err := d.run(); err == nil || !strings.Contains(err.Error(), "empty memo") {
		t.Fatalf("LONG_BINPUT error = %v", err)
	}
	d = decoder{data: []byte{'K', 7, 'r', 0, 1, 0, 0, 'j', 0, 1, 0, 0, '.'}, memo: map[int]any{}}
	if value, err := d.run(); err != nil || value != 7 {
		t.Fatalf("valid LONG_BINPUT/GET: value=%v err=%v", value, err)
	}
}

func FuzzBoundedPickleDecode(f *testing.F) {
	for _, input := range [][]byte{{'r', 0, 0, 0, 0}, {'q', 0}, {'R'}, {'('}, {'N', '.'}, {'}', '(', 'u', '.'}, {'c'}} {
		f.Add(input)
	}
	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) > 256 {
			t.Skip()
		}
		// No storage files: tensor rebuilds fail before allocating tensor data.
		d := decoder{data: input, memo: map[int]any{}}
		_, _ = d.run()
	})
}
