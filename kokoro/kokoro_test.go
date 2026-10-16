package kokoro

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/torchrng"
)

func fixtureFloats(t *testing.T, name string) []float32 {
	t.Helper()
	b, e := os.ReadFile(name)
	if e != nil {
		t.Fatal(e)
	}
	if len(b)%4 != 0 {
		t.Fatal("unaligned fixture")
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}
func difference(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("length mismatch")
	}
	var maxAbs float32
	for i, v := range a {
		d := float32(math.Abs(float64(v - b[i])))
		if d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

func TestOfficialSynthesis(t *testing.T) {
	pth := os.Getenv("KOKORO_PTH")
	voicePath := os.Getenv("KOKORO_VOICE")
	if pth == "" || voicePath == "" {
		t.Skip("set KOKORO_PTH and KOKORO_VOICE")
	}
	m, e := Load(pth, filepath.Join(filepath.Dir(pth), "config.json"))
	if e != nil {
		t.Fatal(e)
	}
	voice, e := LoadVoice(voicePath)
	if e != nil {
		t.Fatal(e)
	}
	phonemes := "hˈɛloʊ wˈɜɹld, ðɪs ɪz ə tˈɛst."
	var expectedTokens []int
	b, e := os.ReadFile("testdata/tokens.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &expectedTokens); e != nil {
		t.Fatal(e)
	}
	gotTokens := m.Tokens(phonemes)
	if len(gotTokens) != len(expectedTokens) {
		t.Fatalf("tokens %v want %v", gotTokens, expectedTokens)
	}
	for i, v := range gotTokens {
		if v != expectedTokens[i] {
			t.Fatalf("token %d = %d want %d", i, v, expectedTokens[i])
		}
	}
	start := time.Now()
	result, e := m.Synthesize(phonemes, voice, 1, torchrng.New(42))
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("2.375 s output on Go CPU: %s", time.Since(start))
	b, e = os.ReadFile("testdata/durations.json")
	if e != nil {
		t.Fatal(e)
	}
	var durations []int
	if e = json.Unmarshal(b, &durations); e != nil {
		t.Fatal(e)
	}
	if len(result.Durations) != len(durations) {
		t.Fatalf("duration count %d want %d", len(result.Durations), len(durations))
	}
	for i, v := range result.Durations {
		if v != durations[i] {
			t.Errorf("duration %d = %d want %d", i, v, durations[i])
		}
	}
	wantF0 := fixtureFloats(t, "testdata/f0.f32")
	f0Err := difference(result.F0, wantF0)
	var voicedRelative float32
	for i, v := range wantF0 {
		if v > 10 {
			relative := float32(math.Abs(float64((result.F0[i] - v) / v)))
			if relative > voicedRelative {
				voicedRelative = relative
			}
		}
	}
	audioErr := difference(result.Audio, fixtureFloats(t, "testdata/audio.f32"))
	t.Logf("F0 max abs %.6g, voiced max relative %.6g; audio max abs %.6g", f0Err, voicedRelative, audioErr)
	if f0Err > 1e-2 || audioErr > 5e-2 {
		t.Errorf("parity exceeded tolerance")
	}
}

func BenchmarkSynthesize(b *testing.B) {
	pth := os.Getenv("KOKORO_PTH")
	voicePath := os.Getenv("KOKORO_VOICE")
	if pth == "" || voicePath == "" {
		b.Skip("set KOKORO_PTH and KOKORO_VOICE")
	}
	model, err := Load(pth, filepath.Join(filepath.Dir(pth), "config.json"))
	if err != nil {
		b.Fatal(err)
	}
	voice, err := LoadVoice(voicePath)
	if err != nil {
		b.Fatal(err)
	}
	phonemes := "hﾋ伊嬪oﾊ・wﾋ伊慄ｹld, ﾃｰﾉｪs ﾉｪz ﾉ・tﾋ伊孕t."
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err := model.Synthesize(phonemes, voice, 1, torchrng.New(42)); err != nil {
			b.Fatal(err)
		}
	}
}
