package irodori

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yh2237/gograd/autograd"
)

func TestReferenceEncoderRealCheckpointParity(t *testing.T) {
	cache := os.Getenv("HF_HOME")
	if cache == "" {
		t.Skip("set HF_HOME for real checkpoint parity")
	}
	paths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "model.safetensors"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("checkpoint: %v %v", paths, err)
	}
	state, err := autograd.OpenSafeTensorFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	raw, err := os.ReadFile("../testdata/irodori_reference_latent.f32")
	if err != nil {
		t.Fatal(err)
	}
	latent := make([]float32, len(raw)/4)
	for i := range latent {
		latent[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	b, err := os.ReadFile("../testdata/irodori_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		SpeakerState struct {
			Shape     []int     `json:"shape"`
			First     []float32 `json:"first"`
			Last      []float32 `json:"last"`
			Sum       float64   `json:"sum"`
			SquareSum float64   `json:"square_sum"`
		} `json:"speaker_state"`
		SpeakerFirstRows [][]float32 `json:"speaker_first_rows"`
		SpeakerLastRows  [][]float32 `json:"speaker_last_rows"`
	}
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	got, err := (ReferenceEncoder{State: state}).Forward(latent)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != fixture.SpeakerState.Shape[0]*fixture.SpeakerState.Shape[1] {
		t.Fatalf("shape %d", len(got))
	}
	var maxAbs float64
	for i, want := range fixture.SpeakerState.First {
		maxAbs = max(maxAbs, math.Abs(float64(got[i]-want)))
	}
	for i, want := range fixture.SpeakerState.Last {
		maxAbs = max(maxAbs, math.Abs(float64(got[len(got)-len(fixture.SpeakerState.Last)+i]-want)))
	}
	for row, values := range fixture.SpeakerFirstRows {
		for j, want := range values {
			maxAbs = max(maxAbs, math.Abs(float64(got[row*768+j]-want)))
		}
	}
	for row, values := range fixture.SpeakerLastRows {
		for j, want := range values {
			maxAbs = max(maxAbs, math.Abs(float64(got[(len(got)/768-len(fixture.SpeakerLastRows)+row)*768+j]-want)))
		}
	}
	var sum, sq float64
	for _, v := range got {
		sum += float64(v)
		sq += float64(v) * float64(v)
	}
	t.Logf("max_slice_abs=%.9g sum_abs=%.9g square_sum_abs=%.9g go_ms=%.3f", maxAbs, math.Abs(sum-fixture.SpeakerState.Sum), math.Abs(sq-fixture.SpeakerState.SquareSum), float64(time.Since(start).Microseconds())/1000)
	if maxAbs > 0.02 {
		t.Errorf("speaker state parity")
	}
}

func TestReferencePathEightWAVParity(t *testing.T) {
	cache, codecPath, root := os.Getenv("HF_HOME"), os.Getenv("IRODORI_CODEC_SAFE"), os.Getenv("IRODORI_WAVS")
	if cache == "" || codecPath == "" || root == "" {
		t.Skip("set HF_HOME, IRODORI_CODEC_SAFE, IRODORI_WAVS")
	}
	paths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "model.safetensors"))
	if err != nil || len(paths) != 1 {
		t.Fatal(err, paths)
	}
	state, err := autograd.OpenSafeTensorFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	codecState, err := autograd.OpenSafeTensorFile(codecPath)
	if err != nil {
		t.Fatal(err)
	}
	defer codecState.Close()
	var wavPaths []string
	for n := 31; n <= 38; n++ {
		wavPaths = append(wavPaths, filepath.Join(root, fmt.Sprintf("VOICEACTRESS100_%03d.wav", n)))
	}
	start := time.Now()
	latent, err := EncodeReferenceWAVs(DACVAECodec{State: codecState}, wavPaths)
	if err != nil {
		t.Fatal(err)
	}
	codecMS := float64(time.Since(start).Microseconds()) / 1000
	fixtureBytes, err := os.ReadFile("../testdata/irodori_reference_latent.f32")
	if err != nil {
		t.Fatal(err)
	}
	if len(latent)*4 != len(fixtureBytes) {
		t.Fatalf("combined latent size %d", len(latent))
	}
	var latentMax float64
	for i, v := range latent {
		want := math.Float32frombits(binary.LittleEndian.Uint32(fixtureBytes[i*4:]))
		latentMax = max(latentMax, math.Abs(float64(v-want)))
	}
	start = time.Now()
	speaker, err := (ReferenceEncoder{State: state}).Forward(latent)
	if err != nil {
		t.Fatal(err)
	}
	speakerMS := float64(time.Since(start).Microseconds()) / 1000
	b, err := os.ReadFile("../testdata/irodori_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		SpeakerState struct {
			Shape []int     `json:"shape"`
			First []float32 `json:"first"`
			Last  []float32 `json:"last"`
		} `json:"speaker_state"`
	}
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(speaker) != product(fixture.SpeakerState.Shape) {
		t.Fatalf("speaker shape %d", len(speaker))
	}
	var speakerMax float64
	for i, v := range fixture.SpeakerState.First {
		speakerMax = max(speakerMax, math.Abs(float64(speaker[i]-v)))
	}
	for i, v := range fixture.SpeakerState.Last {
		speakerMax = max(speakerMax, math.Abs(float64(speaker[len(speaker)-len(fixture.SpeakerState.Last)+i]-v)))
	}
	t.Logf("latent_full_max_abs=%.9g speaker_slice_max_abs=%.9g go_codec_ms=%.3f go_speaker_ms=%.3f", latentMax, speakerMax, codecMS, speakerMS)
	if latentMax > 0.001 || speakerMax > 0.001 {
		t.Error("reference path parity")
	}
}
