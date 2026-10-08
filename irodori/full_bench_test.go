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

func TestFullSentenceOneStepCPUParity(t *testing.T) {
	if os.Getenv("IRODORI_FULL_BENCH") != "1" {
		t.Skip("opt in with IRODORI_FULL_BENCH=1; encodes eight WAVs and a full sentence")
	}
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
	tokenizerPaths, err := filepath.Glob(filepath.Join(cache, "hub", "models--Aratako--Irodori-TTS-v4.1-Small", "snapshots", "*", "tokenizer", "tokenizer.json"))
	if err != nil || len(tokenizerPaths) != 1 {
		t.Fatal(err, tokenizerPaths)
	}
	tok, err := LoadTokenizer(tokenizerPaths[0])
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../testdata/irodori_integrated.json")
	if err != nil {
		t.Fatal(err)
	}
	var prompt struct{ Text, Caption string }
	if err := json.Unmarshal(b, &prompt); err != nil {
		t.Fatal(err)
	}
	b, err = os.ReadFile("../testdata/irodori_full_one_step.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		LatentSteps int                             `json:"latent_steps"`
		WaveSamples int                             `json:"wave_samples"`
		CodecMS     float64                         `json:"codec_ms"`
		SampleMS    float64                         `json:"sample_ms"`
		DecodeMS    float64                         `json:"decode_ms"`
		FinalLatent struct{ First, Last []float32 } `json:"final_latent"`
		WaveIndices []int                           `json:"wave_indices"`
		WaveValues  []float32                       `json:"wave_values"`
	}
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	var wavPaths []string
	for n := 31; n <= 38; n++ {
		wavPaths = append(wavPaths, filepath.Join(root, fmt.Sprintf("VOICEACTRESS100_%03d.wav", n)))
	}
	codec := DACVAECodec{State: codecState}
	start := time.Now()
	reference, err := EncodeReferenceWAVs(codec, wavPaths)
	if err != nil {
		t.Fatal(err)
	}
	codecMS := float64(time.Since(start).Microseconds()) / 1000
	conditioner := Conditioner{State: state, Tokenizer: tok}
	start = time.Now()
	condition, err := conditioner.Encode(prompt.Text, prompt.Caption, reference, 24)
	if err != nil {
		t.Fatal(err)
	}
	conditionMS := float64(time.Since(start).Microseconds()) / 1000
	duration, err := conditioner.PredictDuration(condition)
	if err != nil {
		t.Fatal(err)
	}
	steps := int(math.Round(math.Expm1(float64(duration))))
	if steps != fixture.LatentSteps {
		t.Fatalf("duration steps %d want %d", steps, fixture.LatentSteps)
	}
	noiseBytes, err := os.ReadFile("../testdata/irodori_full_noise.f32")
	if err != nil {
		t.Fatal(err)
	}
	noise := make([]float32, len(noiseBytes)/4)
	for i := range noise {
		noise[i] = math.Float32frombits(binary.LittleEndian.Uint32(noiseBytes[i*4:]))
	}
	if len(noise) != steps*32 {
		t.Fatalf("noise length %d", len(noise))
	}
	start = time.Now()
	if err := PreloadDenoiser(state); err != nil {
		t.Fatal(err)
	}
	preloadMS := float64(time.Since(start).Microseconds()) / 1000
	start = time.Now()
	useCUDA := os.Getenv("IRODORI_FULL_CUDA_BENCH") == "1"
	if useCUDA {
		ctx, err := autograd.NewCUDAContext()
		if err != nil {
			t.Skip(err)
		}
		defer ctx.Close()
	}
	final, err := conditioner.SampleEulerRF(noise, condition, SamplerConfig{Steps: 1, Schedule: "linear", GuidanceMode: "independent", TextScale: 3, SpeakerScale: 5, CaptionScale: 3, CFGMinT: .5, CFGMaxT: 1, UseCUDA: useCUDA})
	if err != nil {
		t.Fatal(err)
	}
	sampleMS := float64(time.Since(start).Microseconds()) / 1000
	var latentMax float64
	for i, want := range fixture.FinalLatent.First {
		latentMax = max(latentMax, math.Abs(float64(final[i]-want)))
	}
	for i, want := range fixture.FinalLatent.Last {
		latentMax = max(latentMax, math.Abs(float64(final[len(final)-len(fixture.FinalLatent.Last)+i]-want)))
	}
	start = time.Now()
	wave, err := codec.Decode(final)
	if err != nil {
		t.Fatal(err)
	}
	decodeMS := float64(time.Since(start).Microseconds()) / 1000
	if len(wave) != fixture.WaveSamples {
		t.Fatalf("wave samples %d want %d", len(wave), fixture.WaveSamples)
	}
	var waveMax float64
	for i, at := range fixture.WaveIndices {
		waveMax = max(waveMax, math.Abs(float64(wave[at]-fixture.WaveValues[i])))
	}
	t.Logf("cuda_projections=%t latent_slice_max_abs=%.9g wave_sample_max_abs=%.9g go_codec_ms=%.3f torch_codec_ms=%.3f go_condition_ms=%.3f go_preload_ms=%.3f go_sample_ms=%.3f torch_sample_ms=%.3f go_decode_ms=%.3f torch_decode_ms=%.3f cached_bytes=%d", useCUDA, latentMax, waveMax, codecMS, fixture.CodecMS, conditionMS, preloadMS, sampleMS, fixture.SampleMS, decodeMS, fixture.DecodeMS, state.CachedBytes())
	if latentMax > 0.02 || waveMax > 0.05 {
		t.Error("full sentence parity")
	}
}
