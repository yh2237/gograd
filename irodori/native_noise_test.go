package irodori

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/yh2237/gograd/torchrng"
)

func torchrngNew(seed uint64) *torchrng.Generator { return torchrng.New(seed) }

// readFixtureNoise loads one of the Python-seeded noise fixtures the parity
// tests inject, so the native generator can be checked against it.
func readFixtureNoise(t *testing.T, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile("../testdata/" + name)
	if err != nil {
		t.Skip(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// TestNativeNoiseMatchesFixture checks that torchrng reproduces the reference
// generator for the exact shapes and seeds the parity fixtures inject, which is
// what makes SampleSeeded usable without Python.
func TestNativeNoiseMatchesFixture(t *testing.T) {
	for _, c := range []struct {
		file  string
		frame int
		seed  uint64
	}{
		{"irodori_sampler_noise.f32", 4, 7231},
		{"irodori_full_noise.f32", 116, 7231},
	} {
		want := readFixtureNoise(t, c.file)
		if len(want) != c.frame*32 {
			t.Fatalf("%s has %d elements, want %d", c.file, len(want), c.frame*32)
		}
		got := make([]float32, c.frame*32)
		torchrngNew(c.seed).Normal(got, 0, 1)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: mismatch at %d (got %v want %v)", c.file, i, got[i], want[i])
			}
		}
		t.Logf("%s: %d elements match the reference seed %d", c.file, len(want), c.seed)
	}
}

// TestSampleSeededMatchesInjectedNoise runs the full sampler twice, once with
// the fixture noise and once with noise drawn from the native generator, and
// requires identical output.
func TestSampleSeededMatchesInjectedNoise(t *testing.T) {
	fixture := openIntegratedFixture(t)
	noise := readFixtureNoise(t, "irodori_sampler_noise.f32")
	injected, err := fixture.Conditioner.SampleEulerRF(noise, fixture.Conditions, SamplerConfig{
		Steps: 3, Schedule: "sway", SwayCoeff: -1, GuidanceMode: "independent",
		TextScale: 3, SpeakerScale: 5, CaptionScale: 3, CFGMinT: .5, CFGMaxT: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	seeded, err := fixture.Conditioner.SampleSeeded(fixture.Conditions, SamplerConfig{
		Steps: 3, Schedule: "sway", SwayCoeff: -1, GuidanceMode: "independent",
		TextScale: 3, SpeakerScale: 5, CaptionScale: 3, CFGMinT: .5, CFGMaxT: 1,
	}, len(noise)/32, 7231)
	if err != nil {
		t.Fatal(err)
	}
	if len(seeded) != len(injected) {
		t.Fatalf("length %d != %d", len(seeded), len(injected))
	}
	for i := range injected {
		if math.Float32bits(seeded[i]) != math.Float32bits(injected[i]) {
			t.Fatalf("seeded sampler differs at %d: %v want %v", i, seeded[i], injected[i])
		}
	}
}
