package data

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestWAVPCMEndpointQuantization(t *testing.T) {
	samples := []float32{-math.MaxFloat32, -1, math.Nextafter32(-1, 0), -.5, 0, .5, math.Nextafter32(1, 0), 1, math.MaxFloat32}
	for _, bits := range []int{8, 16, 24, 32} {
		t.Run(fmt.Sprint(bits), func(t *testing.T) {
			var buf bytes.Buffer
			if err := WriteWAV(&buf, samples, WAVEncoding{Channels: 1, SampleRate: 8000, Bits: bits}); err != nil {
				t.Fatal(err)
			}
			payload := buf.Bytes()[44:]
			scale := float64((int64(1) << uint(bits-1)) - 1)
			for i, sample := range samples {
				raw := payload[i*(bits/8):]
				var got int64
				switch bits {
				case 8:
					got = int64(raw[0]) - 128
				case 16:
					got = int64(int16(binary.LittleEndian.Uint16(raw)))
				case 24:
					q := int32(raw[0]) | int32(raw[1])<<8 | int32(raw[2])<<16
					got = int64(q << 8 >> 8)
				case 32:
					got = int64(int32(binary.LittleEndian.Uint32(raw)))
				}
				want := int64(math.Round(math.Max(-1, math.Min(1, float64(sample))) * scale))
				if got != want {
					t.Fatalf("sample %g: raw PCM=%d want %d", sample, got, want)
				}
			}
		})
	}
}

func TestReadPalettedPNGChannels(t *testing.T) {
	for _, tc := range []struct {
		name    string
		palette color.Palette
		shape   []int
		values  []float32
	}{
		{"color", color.Palette{color.RGBA{R: 255, A: 255}, color.RGBA{G: 255, A: 255}}, []int{3, 1, 2}, []float32{1, 0, 0, 1, 0, 0}},
		{"gray", color.Palette{color.Gray{Y: 64}, color.Gray{Y: 192}}, []int{1, 1, 2}, []float32{64.0 / 255, 192.0 / 255}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			img := image.NewPaletted(image.Rect(0, 0, 2, 1), tc.palette)
			img.Pix[1] = 1
			var buf bytes.Buffer
			if err := png.Encode(&buf, img); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "palette.png")
			if err := os.WriteFile(path, buf.Bytes(), 0600); err != nil {
				t.Fatal(err)
			}
			values, shape, err := ReadImageFile(path)
			if err != nil || !slices.Equal(shape, tc.shape) || !slices.Equal(values, tc.values) {
				t.Fatalf("shape=%v values=%v err=%v", shape, values, err)
			}
		})
	}
}

func TestScaleImageGrayLuminanceAndOrigin(t *testing.T) {
	img := image.NewRGBA(image.Rect(10, 20, 12, 21))
	red, green := color.RGBA{R: 255, A: 255}, color.RGBA{G: 255, A: 255}
	img.Set(10, 20, red)
	img.Set(11, 20, green)
	want := []uint8{color.GrayModel.Convert(red).(color.Gray).Y, color.GrayModel.Convert(green).(color.Gray).Y}
	if got := ScaleImage(img, 2, 1, 1); !slices.Equal(got, want) {
		t.Fatalf("gray=%v want %v", got, want)
	}
	if got := ScaleImage(img, 2, 1, 3); !slices.Equal(got, []uint8{255, 0, 0, 255, 0, 0}) {
		t.Fatalf("RGB=%v", got)
	}
}
