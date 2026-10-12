package data

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func monoSamples(frames int) []float32 {
	out := make([]float32, frames)
	for i := range out {
		out[i] = float32(i%17-8) / 9
	}
	return out
}

func TestWAVRoundTrip(t *testing.T) {
	for _, encoding := range []WAVEncoding{
		{Channels: 1, SampleRate: 16000, Bits: 16},
		{Channels: 2, SampleRate: 44100, Bits: 8},
		{Channels: 1, SampleRate: 8000, Bits: 24},
		{Channels: 1, SampleRate: 8000, Bits: 32},
		{Channels: 2, SampleRate: 22050, Bits: 32, Float: true},
		{Channels: 1, SampleRate: 22050, Bits: 64, Float: true},
	} {
		samples := monoSamples(64 * encoding.Channels)
		var buffer bytes.Buffer
		if err := WriteWAV(&buffer, samples, encoding); err != nil {
			t.Fatalf("%+v: %v", encoding, err)
		}
		wav, err := ReadWAV(&buffer)
		if err != nil {
			t.Fatalf("%+v: %v", encoding, err)
		}
		if wav.SampleRate != encoding.SampleRate || wav.Channels != encoding.Channels || wav.Bits != encoding.Bits || wav.Float != encoding.Float || wav.Frames() != 64 {
			t.Fatalf("%+v decoded as %+v", encoding, wav)
		}
		// Integer PCM stores q = round(x * (2^(bits-1)-1)) and decodes q / 2^(bits-1),
		// so a round trip is within 1.5 least-significant steps.
		step := float64(1) / float64(uint64(1)<<uint(encoding.Bits-1))
		for i, want := range samples {
			got := wav.Samples[i]
			diff := math.Abs(float64(got - want))
			if !encoding.Float && diff > 1.5*step {
				t.Fatalf("%+v sample %d = %v want %v", encoding, i, got, want)
			}
			if encoding.Float && got != want {
				t.Fatalf("%+v float sample %d = %v want %v", encoding, i, got, want)
			}
		}
	}
}

func TestWAVMonoAndFile(t *testing.T) {
	stereo := []float32{1, -1, 0.5, -0.5, 0, 0}
	var buffer bytes.Buffer
	if err := WriteWAV(&buffer, stereo, WAVEncoding{Channels: 2, SampleRate: 32000, Bits: 32, Float: true}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tone.wav")
	if err := os.WriteFile(path, buffer.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	wav, err := ReadWAV(&buffer)
	if err != nil {
		t.Fatal(err)
	}
	mono := wav.Mono()
	if len(mono) != 3 || mono[0] != 0 || mono[1] != 0 || mono[2] != 0 {
		t.Fatalf("mono %v", mono)
	}
	fileMono, rate, err := ReadWAVMono(path)
	if err != nil || rate != 32000 || len(fileMono) != 3 {
		t.Fatalf("ReadWAVMono %v %d %v", fileMono, rate, err)
	}
	if _, _, err := ReadWAVMono(filepath.Join(t.TempDir(), "missing.wav")); err == nil {
		t.Fatal("missing file accepted")
	}
}

func TestWAVExtensible(t *testing.T) {
	// Minimal WAVE_FORMAT_EXTENSIBLE 32-bit float container.
	samples := []float32{0.25, -0.5}
	var buffer bytes.Buffer
	buffer.WriteString("RIFF")
	binary.Write(&buffer, binary.LittleEndian, uint32(0))
	buffer.WriteString("WAVE")
	var fmtChunk bytes.Buffer
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(waveFormatExtensible))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(1))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(48000))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(4))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(4))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(32))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(22))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(32))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(0))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(waveFormatFloat))
	fmtChunk.Write(make([]byte, 12))
	buffer.WriteString("fmt ")
	binary.Write(&buffer, binary.LittleEndian, uint32(fmtChunk.Len()))
	buffer.Write(fmtChunk.Bytes())
	buffer.WriteString("data")
	binary.Write(&buffer, binary.LittleEndian, uint32(len(samples)*4))
	for _, value := range samples {
		binary.Write(&buffer, binary.LittleEndian, math.Float32bits(value))
	}
	wav, err := ReadWAV(&buffer)
	if err != nil {
		t.Fatal(err)
	}
	if !wav.Float || wav.Bits != 32 || wav.SampleRate != 48000 || wav.Samples[1] != -0.5 {
		t.Fatalf("extensible %+v", wav)
	}
}

func TestWAVErrors(t *testing.T) {
	valid := func() *bytes.Buffer {
		var buffer bytes.Buffer
		if err := WriteWAV(&buffer, monoSamples(4), WAVEncoding{Channels: 1, SampleRate: 8000, Bits: 16}); err != nil {
			t.Fatal(err)
		}
		return &buffer
	}
	if _, err := ReadWAV(nil); err == nil {
		t.Fatal("nil reader accepted")
	}
	if _, err := ReadWAV(bytes.NewReader([]byte("not a riff"))); err == nil {
		t.Fatal("bad magic accepted")
	}
	raw := valid().Bytes()
	for name, mangled := range map[string][]byte{
		"truncated": raw[:20],
		"fmt":       append(append([]byte{}, raw[:20]...), 0, 0, 0, 0),
		"data":      raw[:44],
	} {
		if _, err := ReadWAV(bytes.NewReader(mangled)); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	// A data chunk that is not a whole number of frames is rejected.
	bad := append([]byte{}, raw...)
	binary.LittleEndian.PutUint32(bad[40:], 6)
	if _, err := ReadWAV(bytes.NewReader(bad)); err == nil {
		t.Fatal("misaligned data accepted")
	}
}

func TestWriteWAVValidation(t *testing.T) {
	for name, test := range map[string]struct {
		samples  []float32
		encoding WAVEncoding
	}{
		"channels":   {[]float32{0}, WAVEncoding{Channels: 0, SampleRate: 8000, Bits: 16}},
		"rate":       {[]float32{0}, WAVEncoding{Channels: 1, SampleRate: 0, Bits: 16}},
		"bits":       {[]float32{0}, WAVEncoding{Channels: 1, SampleRate: 8000, Bits: 12}},
		"float bits": {[]float32{0}, WAVEncoding{Channels: 1, SampleRate: 8000, Bits: 16, Float: true}},
		"partial":    {[]float32{0}, WAVEncoding{Channels: 2, SampleRate: 8000, Bits: 16}},
		"nonfinite":  {[]float32{float32(math.Inf(1))}, WAVEncoding{Channels: 1, SampleRate: 8000, Bits: 16}},
	} {
		if err := WriteWAV(&bytes.Buffer{}, test.samples, test.encoding); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if err := WriteWAV(nil, []float32{0}, WAVEncoding{Channels: 1, SampleRate: 8000, Bits: 16}); err == nil {
		t.Fatal("nil writer accepted")
	}
}
