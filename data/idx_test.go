package data

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestIDXRawGzipAndValidation(t *testing.T) {
	images := make([]byte, 16+12)
	labels := make([]byte, 8+2)
	for i, value := range []uint32{2051, 2, 2, 3} {
		binary.BigEndian.PutUint32(images[i*4:], value)
	}
	binary.BigEndian.PutUint32(labels, 2049)
	binary.BigEndian.PutUint32(labels[4:], 2)
	copy(images[16:], []byte{0, 51, 102, 153, 204, 255, 5, 10, 15, 20, 25, 30})
	copy(labels[8:], []byte{0, 9})
	dir := t.TempDir()
	write := func(name string, b []byte, compressed bool) string {
		path := filepath.Join(dir, name)
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if compressed {
			z := gzip.NewWriter(f)
			if _, err := z.Write(b); err != nil {
				t.Fatal(err)
			}
			if err := z.Close(); err != nil {
				t.Fatal(err)
			}
		} else if _, err := f.Write(b); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		return path
	}
	rawImages, rawLabels := write("images", images, false), write("labels", labels, false)
	raw, err := OpenIDX(rawImages, rawLabels)
	if err != nil {
		t.Fatal(err)
	}
	compressed, err := OpenIDX(write("images.gz", images, true), write("labels.gz", labels, true))
	if err != nil {
		t.Fatal(err)
	}
	if raw.Fingerprint() != compressed.Fingerprint() || raw.Len() != 2 || raw.Classes() != 10 {
		t.Fatal("IDX identity/labels mismatch")
	}
	sample, err := raw.Get(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(sample.Shape, []int{1, 2, 3}) || !slices.Equal(sample.Values, []float32{0, .2, .4, .6, .8, 1}) {
		t.Fatal("IDX layout or normalization mismatch")
	}
	sample.Values[0] = 17
	sample, _ = raw.Get(context.Background(), 0)
	if sample.Values[0] != 0 {
		t.Fatal("IDX Get modified dataset")
	}
	for name, invalid := range map[string][]byte{"truncated": images[:len(images)-1], "extra": append(slices.Clone(images), 0), "short": []byte{1, 2}} {
		if _, err := OpenIDX(write(name, invalid, false), rawLabels); err == nil {
			t.Fatalf("%s IDX accepted", name)
		}
	}
	claimed := slices.Clone(images)
	binary.BigEndian.PutUint32(claimed[8:], ^uint32(0))
	if _, err := OpenIDX(write("huge_header", claimed, false), rawLabels); err == nil {
		t.Fatal("oversized header accepted")
	}
	wrong := slices.Clone(labels)
	binary.BigEndian.PutUint32(wrong[4:], 3)
	if _, err := OpenIDX(rawImages, write("wrong_labels", wrong, false)); err == nil {
		t.Fatal("label count mismatch accepted")
	}
}
