package data

import (
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
)

// ReadImageFile decodes a PNG or JPEG file into float32 normalized to [0,1].
// The returned shape is [channels,height,width]; RGB images keep 3 channels and
// grayscale images get one.
func ReadImageFile(path string) ([]float32, []int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, nil, fmt.Errorf("data: image decode %s: %w", path, err)
	}
	return pixelsFromImage(img)
}

func pixelsFromImage(img image.Image) ([]float32, []int, error) {
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	switch img.(type) {
	case *image.Gray, *image.Gray16, *image.Paletted:
		out := make([]float32, width*height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, _, _, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
				out[y*width+x] = float32(r) / 65535
			}
		}
		return out, []int{1, height, width}, nil
	default:
		out := make([]float32, 3*width*height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, blue, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
				out[0*width*height+(y*width)+x] = float32(r) / 65535
				out[1*width*height+(y*width)+x] = float32(g) / 65535
				out[2*width*height+(y*width)+x] = float32(blue) / 65535
			}
		}
		return out, []int{3, height, width}, nil
	}
}

// WriteIDX stores images and labels as MNIST-compatible IDX files.
func WriteIDX(imagesPath, labelsPath string, rows, cols int, images [][]byte, labels []byte) error {
	if rows <= 0 || cols <= 0 || len(labels) != len(images) {
		return fmt.Errorf("data: inconsistent IDX payload")
	}
	imageBytes := make([]byte, 16+len(images)*rows*cols)
	binary.BigEndian.PutUint32(imageBytes, 2051)
	binary.BigEndian.PutUint32(imageBytes[4:], uint32(len(images)))
	binary.BigEndian.PutUint32(imageBytes[8:], uint32(rows))
	binary.BigEndian.PutUint32(imageBytes[12:], uint32(cols))
	offset := 16
	for _, image := range images {
		if len(image) != rows*cols {
			return fmt.Errorf("data: image has %d bytes, want %d", len(image), rows*cols)
		}
		copy(imageBytes[offset:], image)
		offset += rows * cols
	}
	labelBytes := make([]byte, 8+len(labels))
	binary.BigEndian.PutUint32(labelBytes, 2049)
	binary.BigEndian.PutUint32(labelBytes[4:], uint32(len(labels)))
	copy(labelBytes[8:], labels)

	if err := os.MkdirAll(filepath.Dir(imagesPath), 0o755); err != nil && filepath.Dir(imagesPath) != "." {
		return err
	}
	if err := os.WriteFile(imagesPath, imageBytes, 0o644); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(labelsPath), 0o755); err != nil && filepath.Dir(labelsPath) != "." {
		return err
	}
	return os.WriteFile(labelsPath, labelBytes, 0o644)
}

// ScaleImage resamples img to width/height by nearest neighbor. If channels is 3,
// the result has 3 planes; otherwise a single grayscale plane.
func ScaleImage(img image.Image, width, height, channels int) []uint8 {
	srcH, srcW := img.Bounds().Dy(), img.Bounds().Dx()
	if channels == 3 {
		out := make([]uint8, 3*width*height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(srcW*x/width, srcH*y/height).RGBA()
				out[0*width*height+(y*width)+x] = uint8(r >> 8)
				out[1*width*height+(y*width)+x] = uint8(g >> 8)
				out[2*width*height+(y*width)+x] = uint8(b >> 8)
			}
		}
		return out
	}
	out := make([]uint8, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, _, _, _ := img.At(srcW*x/width, srcH*y/height).RGBA()
			out[y*width+x] = uint8(r >> 8)
		}
	}
	return out
}
