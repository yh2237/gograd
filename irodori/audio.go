package irodori

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/yh2237/gograd/dsp"
)

// ReadWAVMono reads uncompressed PCM16 or F32 RIFF audio and averages channels.
// It returns samples in the same float32 range as torchaudio.load.
func ReadWAVMono(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	var header [12]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, 0, err
	}
	if string(header[:4]) != "RIFF" || string(header[8:]) != "WAVE" {
		return nil, 0, fmt.Errorf("irodori: unsupported WAV container")
	}
	var format uint16
	var channels uint16
	var sampleRate uint32
	var bits uint16
	var data []byte
	for {
		var chunk [8]byte
		if _, err := io.ReadFull(f, chunk[:]); err == io.EOF {
			break
		} else if err != nil {
			return nil, 0, err
		}
		size := binary.LittleEndian.Uint32(chunk[4:])
		if size > 1<<30 {
			return nil, 0, fmt.Errorf("irodori: oversized WAV chunk")
		}
		payload := make([]byte, int(size))
		if _, err := io.ReadFull(f, payload); err != nil {
			return nil, 0, err
		}
		if size%2 != 0 {
			if _, err := f.Seek(1, io.SeekCurrent); err != nil {
				return nil, 0, err
			}
		}
		switch string(chunk[:4]) {
		case "fmt ":
			if len(payload) < 16 {
				return nil, 0, fmt.Errorf("irodori: short WAV fmt")
			}
			format = binary.LittleEndian.Uint16(payload)
			channels = binary.LittleEndian.Uint16(payload[2:])
			sampleRate = binary.LittleEndian.Uint32(payload[4:])
			bits = binary.LittleEndian.Uint16(payload[14:])
		case "data":
			data = payload
		}
	}
	if channels == 0 || sampleRate == 0 || len(data) == 0 {
		return nil, 0, fmt.Errorf("irodori: missing WAV fmt/data")
	}
	bytesPerSample := int(bits / 8)
	if !((format == 1 && bits == 16) || (format == 3 && bits == 32)) || len(data)%(int(channels)*bytesPerSample) != 0 {
		return nil, 0, fmt.Errorf("irodori: unsupported WAV format %d/%d", format, bits)
	}
	frames := len(data) / (int(channels) * bytesPerSample)
	out := make([]float32, frames)
	for frame := 0; frame < frames; frame++ {
		var sum float32
		for channel := 0; channel < int(channels); channel++ {
			at := (frame*int(channels) + channel) * bytesPerSample
			if format == 1 {
				sum += float32(int16(binary.LittleEndian.Uint16(data[at:]))) / 32768
			} else {
				sum += math.Float32frombits(binary.LittleEndian.Uint32(data[at:]))
			}
		}
		out[frame] = sum / float32(channels)
	}
	return out, int(sampleRate), nil
}

var kWeighting48k = [2]struct{ a, b [3]float32 }{
	{a: [3]float32{1, -1.6906995865986896, 0.7325047060963897}, b: [3]float32{1.5351828863637502, -2.691804030199196, 1.198426263333146}},
	{a: [3]float32{1, -1.990076284018423, 0.9901009040531438}, b: [3]float32{0.9950442970178917, -1.9900885940357833, 0.9950442970178917}},
}

// NormalizeReference48k follows AudioTools' K-weighted 400 ms/75%-overlap
// gated LUFS normalization and peak safety scale used by the Irodori codec.
func NormalizeReference48k(input []float32) ([]float32, error) {
	if len(input) < 19200 {
		return nil, fmt.Errorf("irodori: reference shorter than loudness block")
	}
	filtered := append([]float32(nil), input...)
	for _, stage := range kWeighting48k {
		var x1, x2, y1, y2 float32
		for i, x := range filtered {
			y := stage.b[0]*x + stage.b[1]*x1 + stage.b[2]*x2 - stage.a[1]*y1 - stage.a[2]*y2
			filtered[i] = y
			x2, x1 = x1, x
			y2, y1 = y1, y
		}
	}
	const block, stride = 19200, 4800
	var powers []float32
	blocks := 1
	if len(filtered) > block {
		blocks += (len(filtered) - block + stride - 1) / stride
	}
	for frame := 0; frame < blocks; frame++ {
		start := frame * stride
		end := min(start+block, len(filtered))
		var sum float32
		for _, v := range filtered[start:end] {
			sum += v * v
		}
		powers = append(powers, sum/block)
	}
	var firstSum float32
	firstCount := 0
	for _, z := range powers {
		if z > 0 && -0.691+10*math.Log10(float64(z)) > -70 {
			firstSum += z
			firstCount++
		}
	}
	if firstCount == 0 {
		return nil, fmt.Errorf("irodori: silent reference")
	}
	threshold := -0.691 + 10*math.Log10(float64(firstSum/float32(firstCount))) - 10
	var gatedSum float32
	gatedCount := 0
	for _, z := range powers {
		if z > 0 {
			db := -0.691 + 10*math.Log10(float64(z))
			if db > -70 && db > threshold {
				gatedSum += z
				gatedCount++
			}
		}
	}
	if gatedCount == 0 {
		return nil, fmt.Errorf("irodori: no gated loudness blocks")
	}
	loudness := -0.691 + 10*math.Log10(float64(gatedSum/float32(gatedCount)))
	if loudness < -70 {
		loudness = -70
	}
	gain := float32(math.Exp((-16 - loudness) * 0.11512925464970229))
	out := make([]float32, len(input))
	var peak float32
	for i, v := range input {
		out[i] = v * gain
		peak = max(peak, float32(math.Abs(float64(out[i]))))
	}
	if peak > 1 {
		for i := range out {
			out[i] /= peak
		}
	}
	return out, nil
}

// ResampleTo48k follows torchaudio.functional.resample's default F32
// sinc_interp_hann kernel, including GCD-reduced phases and zero padding.
// ResampleTo48k resamples a mono block to 48 kHz, the rate the codec, the
// reference loudness normalization and the watermark all operate at.
func ResampleTo48k(input []float32, sampleRate int) ([]float32, error) {
	if sampleRate <= 0 || len(input) == 0 {
		return nil, fmt.Errorf("irodori: invalid resampling input")
	}
	return dsp.Resample(input, sampleRate, 48000), nil
}
