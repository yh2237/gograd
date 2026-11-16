package data

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func wavBytes(sampleRate, channels, bits int, payload []byte) []byte {
	b := make([]byte, 0, 44+len(payload))
	b = append(b, "RIFF"...)
	b = binary.LittleEndian.AppendUint32(b, uint32(36+len(payload)))
	b = append(b, "WAVE"...)
	b = append(b, "fmt "...)
	b = binary.LittleEndian.AppendUint32(b, 16)
	b = binary.LittleEndian.AppendUint16(b, 1)
	b = binary.LittleEndian.AppendUint16(b, uint16(channels))
	b = binary.LittleEndian.AppendUint32(b, uint32(sampleRate))
	b = binary.LittleEndian.AppendUint32(b, uint32(sampleRate*channels*bits/8))
	b = binary.LittleEndian.AppendUint16(b, uint16(channels*bits/8))
	b = binary.LittleEndian.AppendUint16(b, uint16(bits))
	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(payload)))
	return append(b, payload...)
}

func TestReadWAVFile(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x40, 0x00, 0xC0}
	path := filepath.Join(t.TempDir(), "tone.wav")
	if err := os.WriteFile(path, wavBytes(8000, 1, 16, payload), 0o600); err != nil {
		t.Fatal(err)
	}
	wav, err := ReadWAVFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if wav.SampleRate != 8000 || wav.Channels != 1 || wav.Bits != 16 || wav.Float {
		t.Fatalf("header = %d Hz / %d ch / %d bit / float=%v", wav.SampleRate, wav.Channels, wav.Bits, wav.Float)
	}
	want := []float32{0, .5, -.5}
	if len(wav.Samples) != len(want) {
		t.Fatalf("samples = %d, want %d", len(wav.Samples), len(want))
	}
	for i := range want {
		if wav.Samples[i] != want[i] {
			t.Fatalf("samples[%d] = %g, want %g", i, wav.Samples[i], want[i])
		}
	}
	if wav.Frames() != 3 {
		t.Fatalf("frames = %d, want 3", wav.Frames())
	}
}

func TestReadWAVFileMissing(t *testing.T) {
	if _, err := ReadWAVFile(filepath.Join(t.TempDir(), "absent.wav")); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func TestReadWAVFileRejectsTruncated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "short.wav")
	if err := os.WriteFile(path, []byte("RIFF"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadWAVFile(path); err == nil {
		t.Fatal("expected an error for a truncated file")
	}
}
