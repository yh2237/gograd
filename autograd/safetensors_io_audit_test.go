package autograd

import (
	encodingbinary "encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func writeIOAuditSafeFixture(t *testing.T, header map[string]safeHeader, payload []byte) string {
	t.Helper()
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := make([]byte, 8+len(h)+len(payload))
	encodingbinary.LittleEndian.PutUint64(file, uint64(len(h)))
	copy(file[8:], h)
	copy(file[8+len(h):], payload)
	path := filepath.Join(t.TempDir(), "fixture.safetensors")
	if err := os.WriteFile(path, file, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSafetensorsIOAuditOverlapBeforeAllocation(t *testing.T) {
	const count = 16 * 1024
	header := make(map[string]safeHeader)
	for i := 0; i < 256; i++ {
		header[fmt.Sprintf("alias%d", i)] = safeHeader{"F32", []int{count}, [2]int{0, count * 4}}
	}
	path := writeIOAuditSafeFixture(t, header, make([]byte, count*4))
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	entries, _, err := readSafetensors(path)
	runtime.ReadMemStats(&after)
	if err == nil || !strings.Contains(err.Error(), "overlapping") || entries != nil {
		t.Fatalf("entries=%d err=%v", len(entries), err)
	}
	// Decoding every alias would allocate 16 MiB from a 64 KiB payload.
	// Allow ample room for reading and validating the JSON header, but not
	// even a quarter of that per-tensor amplification.
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated >= 4<<20 {
		t.Fatalf("overlapping payload allocated %d bytes before rejection", allocated)
	}
}

func TestSafetensorsIOAuditLayout(t *testing.T) {
	for name, meta := range map[string]safeHeader{
		"negative": {"F32", []int{1}, [2]int{-1, 3}},
		"reversed": {"F32", []int{1}, [2]int{4, 0}},
		"outside":  {"F32", []int{1}, [2]int{4, 8}},
		"shape":    {"F32", []int{2}, [2]int{0, 4}},
		"dtype":    {"F16", []int{1}, [2]int{0, 4}},
	} {
		t.Run(name, func(t *testing.T) {
			path := writeIOAuditSafeFixture(t, map[string]safeHeader{"bad": meta}, make([]byte, 4))
			if _, _, err := readSafetensors(path); err == nil {
				t.Fatal("invalid tensor accepted")
			}
		})
	}
	payload := make([]byte, 8)
	encodingbinary.LittleEndian.PutUint32(payload, math.Float32bits(1.25))
	encodingbinary.LittleEndian.PutUint32(payload[4:], math.Float32bits(-2.5))
	path := writeIOAuditSafeFixture(t, map[string]safeHeader{
		"a":     {"F32", []int{1}, [2]int{0, 4}},
		"b":     {"F32", []int{1}, [2]int{4, 8}},
		"empty": {"F32", []int{0}, [2]int{4, 4}},
	}, payload)
	entries, _, err := readSafetensors(path)
	if err != nil || entries["a"].Values[0] != 1.25 || entries["b"].Values[0] != -2.5 || len(entries["empty"].Values) != 0 {
		t.Fatalf("entries=%v err=%v", entries, err)
	}
}
