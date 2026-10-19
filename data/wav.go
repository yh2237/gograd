package data

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// WAV holds decoded uncompressed RIFF audio. Samples are interleaved by channel
// and normalized the way torchaudio/PyTorch loads them: integer PCM is divided by
// 2^(bits-1), so positive full scale stays below 1, while IEEE float data is kept
// as written.
type WAV struct {
	SampleRate int
	Channels   int
	Bits       int
	Float      bool
	Samples    []float32
}

// Frames is the number of sample frames (one value per channel).
func (w *WAV) Frames() int {
	if w == nil || w.Channels <= 0 {
		return 0
	}
	return len(w.Samples) / w.Channels
}

// Mono averages every channel into one signal, matching the common
// single-channel analysis path.
func (w *WAV) Mono() []float32 {
	if w == nil || w.Channels <= 0 {
		return nil
	}
	if w.Channels == 1 {
		return append([]float32(nil), w.Samples...)
	}
	out := make([]float32, w.Frames())
	for frame := range out {
		var sum float32
		base := frame * w.Channels
		for channel := 0; channel < w.Channels; channel++ {
			sum += w.Samples[base+channel]
		}
		out[frame] = sum / float32(w.Channels)
	}
	return out
}

const (
	waveFormatPCM        = 1
	waveFormatFloat      = 3
	waveFormatExtensible = 0xfffe
)

// ReadWAV decodes an uncompressed PCM or IEEE-float RIFF file. It accepts 8/16/
// 24/32-bit integer PCM (8-bit is unsigned, as the format requires) and 32/64-bit
// float, including WAVE_FORMAT_EXTENSIBLE wrappers. Compressed codecs and
// unknown/unparseable layouts are rejected rather than guessed at.
func ReadWAV(r io.Reader) (*WAV, error) {
	if r == nil {
		return nil, fmt.Errorf("data: nil WAV reader")
	}
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, fmt.Errorf("data: short WAV header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, fmt.Errorf("data: not a RIFF/WAVE stream")
	}
	var format, channels, bits int
	var sampleRate int
	var payload []byte
	for {
		var header [8]byte
		if _, err := io.ReadFull(r, header[:]); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("data: truncated WAV chunk: %w", err)
		}
		size := int(binary.LittleEndian.Uint32(header[4:]))
		if size < 0 || size > 1<<30 {
			return nil, fmt.Errorf("data: invalid WAV chunk size %d", size)
		}
		body := make([]byte, size)
		if _, err := io.ReadFull(r, body); err != nil {
			return nil, fmt.Errorf("data: truncated WAV chunk %q: %w", header[0:4], err)
		}
		if size%2 != 0 {
			var padding [1]byte
			if _, err := io.ReadFull(r, padding[:]); err != nil && err != io.EOF {
				return nil, err
			}
		}
		switch string(header[0:4]) {
		case "fmt ":
			if len(body) < 16 {
				return nil, fmt.Errorf("data: short WAV fmt chunk")
			}
			format = int(binary.LittleEndian.Uint16(body[0:]))
			channels = int(binary.LittleEndian.Uint16(body[2:]))
			sampleRate = int(binary.LittleEndian.Uint32(body[4:]))
			bits = int(binary.LittleEndian.Uint16(body[14:]))
			if format == waveFormatExtensible {
				if len(body) < 40 {
					return nil, fmt.Errorf("data: short WAVE_FORMAT_EXTENSIBLE fmt chunk")
				}
				format = int(binary.LittleEndian.Uint32(body[24:]))
			}
		case "data":
			payload = body
		}
	}
	if channels <= 0 || sampleRate <= 0 {
		return nil, fmt.Errorf("data: WAV is missing a valid fmt chunk")
	}
	if len(payload) == 0 {
		return nil, fmt.Errorf("data: WAV has no data chunk")
	}
	wav := &WAV{SampleRate: sampleRate, Channels: channels, Bits: bits, Float: format == waveFormatFloat}
	width := bits / 8
	switch {
	case format == waveFormatPCM && (bits == 8 || bits == 16 || bits == 24 || bits == 32):
	case format == waveFormatFloat && (bits == 32 || bits == 64):
	default:
		return nil, fmt.Errorf("data: unsupported WAV format %d/%d", format, bits)
	}
	if width <= 0 || len(payload)%(channels*width) != 0 {
		return nil, fmt.Errorf("data: WAV data size does not match %d channel(s) at %d bits", channels, bits)
	}
	wav.Samples = make([]float32, len(payload)/width)
	for i := range wav.Samples {
		wav.Samples[i] = decodeWAVSample(payload[i*width:], bits, format)
	}
	return wav, nil
}

func decodeWAVSample(raw []byte, bits, format int) float32 {
	switch format {
	case waveFormatFloat:
		if bits == 32 {
			return math.Float32frombits(binary.LittleEndian.Uint32(raw))
		}
		return float32(math.Float64frombits(binary.LittleEndian.Uint64(raw)))
	case waveFormatPCM:
		switch bits {
		case 8:
			return (float32(raw[0]) - 128) / 128
		case 16:
			return float32(int16(binary.LittleEndian.Uint16(raw))) / 32768
		case 24:
			value := int32(raw[0]) | int32(raw[1])<<8 | int32(raw[2])<<16
			if value&0x800000 != 0 {
				value |= ^0xffffff
			}
			return float32(value) / 8388608
		default:
			return float32(int32(binary.LittleEndian.Uint32(raw))) / 2147483648
		}
	}
	return 0
}

// ReadWAVFile opens and decodes one file.
func ReadWAVFile(path string) (*WAV, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ReadWAV(file)
}

// ReadWAVMono decodes a file and returns its channel-averaged signal and rate.
func ReadWAVMono(path string) ([]float32, int, error) {
	wav, err := ReadWAVFile(path)
	if err != nil {
		return nil, 0, err
	}
	return wav.Mono(), wav.SampleRate, nil
}

// WAVEncoding selects the output layout for WriteWAV.
type WAVEncoding struct {
	Channels   int
	SampleRate int
	// Bits is 8, 16, 24 or 32 for integer PCM, or 32/64 when Float is set.
	Bits  int
	Float bool
}

// WriteWAV encodes interleaved samples as an uncompressed RIFF file. Integer
// encodings clamp to the representable range; float encodings reject non-finite
// values so a corrupted signal cannot be written silently.
func WriteWAV(w io.Writer, samples []float32, encoding WAVEncoding) error {
	if w == nil {
		return fmt.Errorf("data: nil WAV writer")
	}
	if encoding.Channels <= 0 || encoding.SampleRate <= 0 {
		return fmt.Errorf("data: invalid WAV encoding")
	}
	width := encoding.Bits / 8
	switch {
	case !encoding.Float && (encoding.Bits == 8 || encoding.Bits == 16 || encoding.Bits == 24 || encoding.Bits == 32):
	case encoding.Float && (encoding.Bits == 32 || encoding.Bits == 64):
	default:
		return fmt.Errorf("data: unsupported WAV encoding %d bits float=%v", encoding.Bits, encoding.Float)
	}
	if len(samples)%encoding.Channels != 0 {
		return fmt.Errorf("data: sample count is not a whole number of frames")
	}
	for _, value := range samples {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("data: non-finite WAV sample")
		}
	}
	format := waveFormatPCM
	if encoding.Float {
		format = waveFormatFloat
	}
	dataBytes := len(samples) * width
	padding := dataBytes % 2
	var header [44]byte
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(36+dataBytes+padding))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], uint16(format))
	binary.LittleEndian.PutUint16(header[22:], uint16(encoding.Channels))
	binary.LittleEndian.PutUint32(header[24:], uint32(encoding.SampleRate))
	binary.LittleEndian.PutUint32(header[28:], uint32(encoding.SampleRate*encoding.Channels*width))
	binary.LittleEndian.PutUint16(header[32:], uint16(encoding.Channels*width))
	binary.LittleEndian.PutUint16(header[34:], uint16(encoding.Bits))
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(dataBytes))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	raw := make([]byte, width)
	for _, value := range samples {
		encodeWAVSample(raw, value, encoding.Bits, encoding.Float)
		if _, err := w.Write(raw); err != nil {
			return err
		}
	}
	if padding != 0 {
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
	}
	return nil
}

func encodeWAVSample(raw []byte, value float32, bits int, float bool) {
	if float {
		if bits == 32 {
			binary.LittleEndian.PutUint32(raw, math.Float32bits(value))
		} else {
			binary.LittleEndian.PutUint64(raw, math.Float64bits(float64(value)))
		}
		return
	}
	// Keep multiplication and rounding in float64: MaxInt32 rounds up to
	// 2147483648 in float32, which converts a positive endpoint to a negative int.
	scale := float64((int64(1) << uint(bits-1)) - 1)
	quantized := math.Round(math.Max(-1, math.Min(1, float64(value))) * scale)
	quantized = math.Max(-scale-1, math.Min(scale, quantized))
	q := int32(quantized)
	switch bits {
	case 8:
		raw[0] = byte(q + 128)
	case 16:
		binary.LittleEndian.PutUint16(raw, uint16(int16(q)))
	case 24:
		raw[0], raw[1], raw[2] = byte(q), byte(q>>8), byte(q>>16)
	default:
		binary.LittleEndian.PutUint32(raw, uint32(q))
	}
}
