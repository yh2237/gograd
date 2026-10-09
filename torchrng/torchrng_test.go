package torchrng

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
)

type streamRecord struct {
	Seed   int   `json:"seed"`
	N      int   `json:"n"`
	Offset int64 `json:"offset"`
	Bytes  int   `json:"bytes"`
}

type streamIndex struct {
	Streams []streamRecord     `json:"streams"`
	Words   map[string][]int64 `json:"words"`
}

// loadStreams reads the PyTorch reference fixture. Each record packs four
// contiguous ranges at its offset: randn(float32), rand(float32),
// randn(float64) and rand(float64), in the order the generator writes them.
func loadStreams(t *testing.T) ([]streamRecord, []byte) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/torchrng_streams.json")
	if err != nil {
		t.Skip(err)
	}
	var index streamIndex
	if err := json.Unmarshal(raw, &index); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile("../testdata/torchrng_streams.bin")
	if err != nil {
		t.Fatal(err)
	}
	return index.Streams, blob
}

func readFloat32(b []byte, offset int64, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[offset+int64(i)*4:]))
	}
	return out
}

func readFloat64(b []byte, offset int64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[offset+int64(i)*8:]))
	}
	return out
}

func firstMismatch32(got, want []float32) int {
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return i
		}
	}
	return -1
}

func firstMismatch64(got, want []float64) int {
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			return i
		}
	}
	return -1
}

// relative64 reports the largest relative difference from want. The reference
// evaluates log, sin and cos with the C runtime's libm, which is not bit
// identical to Go's math, so float64 streams may differ by one ulp.
func relative64(got, want []float64) (worst float64, at int) {
	for i := range got {
		d := math.Abs(got[i]-want[i]) / math.Max(math.Abs(want[i]), 1e-300)
		if d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

// TestStreamParity compares whole streams bit for bit. The fixture records each
// tensor drawn from a freshly seeded generator, so a mismatch names the exact
// element where the port diverges.
func TestStreamParity(t *testing.T) {
	streams, blob := loadStreams(t)
	for _, s := range streams {
		if s.Bytes != s.N*24 {
			t.Fatalf("record sizing: %+v", s)
		}
		at := s.Offset
		wantNormal32 := readFloat32(blob, at, s.N)
		at += int64(s.N) * 4
		wantRand32 := readFloat32(blob, at, s.N)
		at += int64(s.N) * 4
		wantNormal64 := readFloat64(blob, at, s.N)
		at += int64(s.N) * 8
		wantRand64 := readFloat64(blob, at, s.N)
		if int64(len(blob)) < at+int64(s.N)*8 {
			t.Fatalf("truncated stream at %+v", s)
		}

		gotNormal32 := make([]float32, s.N)
		New(uint64(s.Seed)).Normal(gotNormal32, 0, 1)
		if i := firstMismatch32(gotNormal32, wantNormal32); i >= 0 {
			t.Errorf("randn(float32) seed=%d n=%d: first mismatch at %d (got %v want %v)",
				s.Seed, s.N, i, gotNormal32[i], wantNormal32[i])
		}

		// The float32 streams are bit-exact because its cephes polynomials
		// are ported verbatim. Float64 goes through libm, which is the one
		// place the reference cannot be reproduced bit for bit. A one-ulp
		// difference in log, sin or cos is amplified by at most a few ulps
		// through the radius and the final multiply, so the bound below is
		// five ulps rather than one.
		gotNormal64 := make([]float64, s.N)
		New(uint64(s.Seed)).Normal64(gotNormal64, 0, 1)
		worst, worstAt := relative64(gotNormal64, wantNormal64)
		if worst > 1e-15 {
			t.Errorf("randn(float64) seed=%d n=%d: relative error %.3g at %d (got %v want %v)",
				s.Seed, s.N, worst, worstAt, gotNormal64[worstAt], wantNormal64[worstAt])
		}

		gotRand32 := make([]float32, s.N)
		New(uint64(s.Seed)).Uniform(gotRand32, 0, 1)
		if i := firstMismatch32(gotRand32, wantRand32); i >= 0 {
			t.Errorf("rand(float32) seed=%d n=%d: first mismatch at %d (got %v want %v)",
				s.Seed, s.N, i, gotRand32[i], wantRand32[i])
		}

		gotRand64 := make([]float64, s.N)
		New(uint64(s.Seed)).Uniform64(gotRand64, 0, 1)
		if i := firstMismatch64(gotRand64, wantRand64); i >= 0 {
			t.Errorf("rand(float64) seed=%d n=%d: first mismatch at %d (got %v want %v)",
				s.Seed, s.N, i, gotRand64[i], wantRand64[i])
		}
	}
}

// TestStreamContinuation checks that two draws from one generator match two
// separately seeded draws, which is what makes the generator usable inside a
// multi-step sampler. Both lengths must be multiples of sixteen: a partial
// block makes the shorter draw recompute a tail block and consume extra words.
func TestStreamContinuation(t *testing.T) {
	streams, blob := loadStreams(t)
	var whole *streamRecord
	for i := range streams {
		if streams[i].N == 128 {
			whole = &streams[i]
			break
		}
	}
	if whole == nil {
		t.Fatal("no 128-element stream recorded")
	}
	want := readFloat32(blob, whole.Offset, 128)
	g := New(uint64(whole.Seed))
	head := make([]float32, 64)
	g.Normal(head, 0, 1)
	more := make([]float32, 64)
	g.Normal(more, 0, 1)
	for i := range head {
		if math.Float32bits(head[i]) != math.Float32bits(want[i]) {
			t.Fatalf("head mismatch at %d: %v want %v", i, head[i], want[i])
		}
	}
	for i := range more {
		if math.Float32bits(more[i]) != math.Float32bits(want[64+i]) {
			t.Fatalf("continuation mismatch at %d: %v want %v", i, more[i], want[64+i])
		}
	}
}

func TestUniformRange(t *testing.T) {
	g := New(99)
	buf := make([]float32, 4096)
	g.Uniform(buf, -1, 3)
	min, max, sum := float64(buf[0]), float64(buf[0]), 0.0
	for _, v := range buf {
		min = math.Min(float64(v), min)
		max = math.Max(float64(v), max)
		sum += float64(v)
	}
	if min < -1 || max >= 3 {
		t.Fatalf("uniform range [%v, %v)", min, max)
	}
	if mean := sum / float64(len(buf)); math.Abs(mean-1) > 0.1 {
		t.Fatalf("uniform mean %v", mean)
	}
}

func TestNormalMoments(t *testing.T) {
	g := New(1234)
	buf := make([]float32, 16384)
	g.Normal(buf, 2, 0.5)
	mean, m2 := moments32(buf)
	if math.Abs(mean-2) > 0.05 || math.Abs(m2-0.25) > 0.02 {
		t.Fatalf("normal mean=%v variance=%v", mean, m2)
	}
}

func moments32(buf []float32) (mean, variance float64) {
	var sum, sumSq float64
	for _, v := range buf {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}
	n := float64(len(buf))
	mean = sum / n
	return mean, sumSq/n - mean*mean
}

// TestSerialPathMoments covers the sub-sixteen-element path, which pairs
// samples differently from the vector layout and cannot reuse it.
func TestSerialPathMoments(t *testing.T) {
	g := New(2024)
	buf := make([]float32, 8)
	var sum, sumSq float64
	for i := 0; i < 4096; i++ {
		g.Normal(buf, 0, 1)
		for _, v := range buf {
			sum += float64(v)
			sumSq += float64(v) * float64(v)
		}
	}
	n := float64(4096 * len(buf))
	mean := sum / n
	variance := sumSq/n - mean*mean
	if math.Abs(mean) > 0.05 || math.Abs(variance-1) > 0.05 {
		t.Fatalf("serial normal mean=%v variance=%v", mean, variance)
	}
}

func TestStateRoundTrip(t *testing.T) {
	g := New(7)
	g.Normal(make([]float32, 64), 0, 1)
	state := g.State()
	tail := make([]float32, 64)
	g.Normal(tail, 0, 1)

	h := New(7)
	h.Normal(make([]float32, 64), 0, 1)
	h.SetState(state)
	restored := make([]float32, 64)
	h.Normal(restored, 0, 1)
	if i := firstMismatch32(restored, tail); i >= 0 {
		t.Fatalf("state round trip mismatch at %d: %v want %v", i, restored[i], tail[i])
	}
}

// BenchmarkNormal measures the cost of one 3712-element float32 draw, the size
// the Irodori sampler uses for a 116-frame latent.
func BenchmarkNormal(b *testing.B) {
	buf := make([]float32, 116*32)
	g := New(7231)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Normal(buf, 0, 1)
	}
}
