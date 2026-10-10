package data

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// IDX is an in-memory byte image/label dataset (MNIST-compatible magic 2051/2049).
// Get allocates one normalized float32 [1,height,width] image; full data stays
// compact uint8. Files can be raw IDX or gzip; no Python/conversion is needed.
type IDX struct {
	images, labels []byte
	rows, cols     int
	fingerprint    string
	classes        int
}

func readIDXFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var reader io.Reader = f
	var signature [2]byte
	if _, err := f.ReadAt(signature[:], 0); err != nil {
		return nil, err
	}
	if signature == [2]byte{0x1f, 0x8b} {
		z, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer z.Close()
		reader = z
	}
	return io.ReadAll(reader)
}
func OpenIDX(imagesPath, labelsPath string) (*IDX, error) {
	images, err := readIDXFile(imagesPath)
	if err != nil {
		return nil, fmt.Errorf("data: IDX images: %w", err)
	}
	labels, err := readIDXFile(labelsPath)
	if err != nil {
		return nil, fmt.Errorf("data: IDX labels: %w", err)
	}
	if len(images) < 16 || len(labels) < 8 || binary.BigEndian.Uint32(images) != 2051 || binary.BigEndian.Uint32(labels) != 2049 {
		return nil, fmt.Errorf("data: invalid IDX headers")
	}
	count, rows, cols := uint64(binary.BigEndian.Uint32(images[4:])), uint64(binary.BigEndian.Uint32(images[8:])), uint64(binary.BigEndian.Uint32(images[12:]))
	// Validate actual lengths before multiplying dimensions or allocating by header.
	if count == 0 || rows == 0 || cols == 0 || count != uint64(binary.BigEndian.Uint32(labels[4:])) || count != uint64(len(labels)-8) || rows > uint64(len(images)-16) || cols > uint64(len(images)-16)/rows || count > uint64(len(images)-16)/(rows*cols) || count*rows*cols != uint64(len(images)-16) {
		return nil, fmt.Errorf("data: inconsistent IDX dimensions or payload")
	}
	h := sha256.New()
	h.Write(images)
	h.Write(labels)
	d := &IDX{images: images[16:], labels: labels[8:], rows: int(rows), cols: int(cols), fingerprint: fmt.Sprintf("%x", h.Sum(nil))}
	for _, label := range d.labels {
		d.classes = max(d.classes, int(label)+1)
	}
	return d, nil
}
func (d *IDX) Len() int              { return len(d.labels) }
func (d *IDX) ImageSize() (int, int) { return d.rows, d.cols }
func (d *IDX) Classes() int          { return d.classes }

// Fingerprint covers both decoded IDX files, independent of gzip container bytes.
func (d *IDX) Fingerprint() string { return d.fingerprint }
func (d *IDX) Get(ctx context.Context, index int) (FloatSample, error) {
	if ctx == nil {
		return FloatSample{}, fmt.Errorf("data: nil context")
	}
	if err := ctx.Err(); err != nil {
		return FloatSample{}, err
	}
	if index < 0 || index >= d.Len() {
		return FloatSample{}, fmt.Errorf("data: IDX index out of range")
	}
	width := d.rows * d.cols
	values := make([]float32, width)
	for i, value := range d.images[index*width : (index+1)*width] {
		values[i] = float32(value) / 255
	}
	return FloatSample{values, []int{1, d.rows, d.cols}, int(d.labels[index])}, nil
}
