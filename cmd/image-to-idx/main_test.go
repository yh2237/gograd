package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yh2237/gograd/data"
)

func TestConvertImagesPreservesBytes(t *testing.T) {
	dir := t.TempDir()
	classDir := filepath.Join(dir, "src", "class")
	if err := os.MkdirAll(classDir, 0700); err != nil {
		t.Fatal(err)
	}
	img := image.NewGray(image.Rect(0, 0, 2, 2))
	copy(img.Pix, []byte{0, 64, 128, 255})
	f, err := os.Create(filepath.Join(classDir, "pixels.png"))
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	images, labels := filepath.Join(dir, "images.idx"), filepath.Join(dir, "labels.idx")
	if err := convertImages(filepath.Join(dir, "src"), images, labels, 2, io.Discard); err != nil {
		t.Fatal(err)
	}
	ds, err := data.OpenIDX(images, labels)
	if err != nil {
		t.Fatal(err)
	}
	sample, err := ds.Get(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []byte{0, 64, 128, 255} {
		if sample.Values[i] != float32(v)/255 {
			t.Fatalf("pixel %d=%g want byte %d", i, sample.Values[i], v)
		}
	}
}

func TestConvertImagesRejectsClassOverflow(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 257; i++ {
		if err := os.Mkdir(filepath.Join(dir, fmt.Sprintf("class%03d", i)), 0700); err != nil {
			t.Fatal(err)
		}
	}
	err := convertImages(dir, filepath.Join(dir, "images"), filepath.Join(dir, "labels"), 2, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "at most 256") {
		t.Fatalf("class overflow: %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "class256")); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewGray(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "class255", "image.png"), buf.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	images, labels := filepath.Join(dir, "images"), filepath.Join(dir, "labels")
	if err := convertImages(dir, images, labels, 2, io.Discard); err != nil {
		t.Fatalf("256 classes should be supported: %v", err)
	}
	ds, err := data.OpenIDX(images, labels)
	if err != nil {
		t.Fatal(err)
	}
	sample, err := ds.Get(context.Background(), 0)
	if err != nil || sample.Label != 255 {
		t.Fatalf("last byte label=%d err=%v", sample.Label, err)
	}
}

func TestConvertImagesReportsInputFailures(t *testing.T) {
	dir := t.TempDir()
	classDir := filepath.Join(dir, "src", "class")
	if err := os.MkdirAll(classDir, 0700); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(classDir, "broken.png")
	if err := os.WriteFile(bad, []byte("broken"), 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "images")
	err := convertImages(filepath.Join(dir, "src"), out, filepath.Join(dir, "labels"), 2, io.Discard)
	if err == nil || !strings.Contains(err.Error(), bad) || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("decode error: %v", err)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("wrote partial output on decode failure: %v", err)
	}
	missing := filepath.Join(dir, "missing")
	err = convertImages(missing, out, filepath.Join(dir, "labels"), 2, io.Discard)
	if err == nil || !strings.Contains(err.Error(), missing) {
		t.Fatalf("directory error: %v", err)
	}
	// An image-extension directory entry that cannot be read as an image must
	// also fail with its path (portable alternative to permission fixtures).
	if err := os.Remove(bad); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "absent.png"), bad); err != nil {
		t.Logf("open-error symlink fixture unavailable: %v", err)
		return
	}
	err = convertImages(filepath.Join(dir, "src"), out, filepath.Join(dir, "labels"), 2, io.Discard)
	if err == nil || !strings.Contains(err.Error(), bad) || !strings.Contains(err.Error(), "open") {
		t.Fatalf("open error: %v", err)
	}
}
