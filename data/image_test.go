package data

import (
	"context"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteIDXRoundTrip(t *testing.T) {
	dir := t.TempDir()
	images := [][]byte{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
	}
	labels := []byte{3, 7}
	if err := WriteIDX(filepath.Join(dir, "images.idx"), filepath.Join(dir, "labels.idx"), 2, 2, images, labels); err != nil {
		t.Fatal(err)
	}
	idx, err := OpenIDX(filepath.Join(dir, "images.idx"), filepath.Join(dir, "labels.idx"))
	if err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 2 || idx.Classes() != 8 {
		t.Fatalf("idx len=%d classes=%d", idx.Len(), idx.Classes())
	}
	for i := range images {
		s, err := idx.Get(context.Background(), i)
		if err != nil {
			t.Fatal(err)
		}
		for j := range images[i] {
			if s.Values[j] != float32(images[i][j])/255 {
				t.Fatalf("sample %d pixel %d = %v", i, j, s.Values[j])
			}
		}
		if s.Label != int(labels[i]) {
			t.Fatalf("label mismatch")
		}
	}
}

func TestReadGrayPNG(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(x, y, color.Gray{Y: uint8((x + y) * 60)})
		}
	}
	tmp, err := os.CreateTemp("", "test*.png")
	if err != nil {
		t.Fatal(err)
	}
	defer tmp.Close()
	if err := png.Encode(tmp, img); err != nil {
		t.Fatal(err)
	}
	values, shape, err := ReadImageFile(tmp.Name())
	if err != nil {
		t.Fatal(err)
	}
	if shape[0] != 1 || shape[1] != 4 || shape[2] != 4 {
		t.Fatalf("shape=%v", shape)
	}
	if len(values) != 16 {
		t.Fatalf("values=%d", len(values))
	}
}
