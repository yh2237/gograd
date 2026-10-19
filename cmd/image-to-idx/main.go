package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yh2237/gograd/data"
)

func main() {
	src := flag.String("src", "", "directory with one subdirectory per class, each containing images")
	outImages := flag.String("out-images", "", "output IDX image file")
	outLabels := flag.String("out-labels", "", "output IDX label file")
	size := flag.Int("size", 32, "target square image size")
	flag.Parse()

	if *src == "" || *outImages == "" || *outLabels == "" {
		fmt.Fprintln(os.Stderr, "image-to-idx: -src, -out-images and -out-labels are required")
		flag.Usage()
		os.Exit(2)
	}
	if *size <= 0 {
		fmt.Fprintln(os.Stderr, "image-to-idx: -size must be positive")
		os.Exit(2)
	}
	if err := convertImages(*src, *outImages, *outLabels, *size, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func convertImages(src, outImages, outLabels string, size int, log io.Writer) error {
	if src == "" || outImages == "" || outLabels == "" || size <= 0 {
		return fmt.Errorf("image-to-idx: source, outputs and positive size required")
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		return fmt.Errorf("image-to-idx: read directory %s: %w", src, err)
	}
	var classes []string
	for _, e := range entries {
		if e.IsDir() {
			classes = append(classes, e.Name())
		}
	}
	sort.Strings(classes)
	if len(classes) == 0 {
		return fmt.Errorf("image-to-idx: no class subdirectories found in %s", src)
	}
	if len(classes) > 256 {
		return fmt.Errorf("image-to-idx: %s has %d classes; IDX byte labels support at most 256", src, len(classes))
	}
	classIndex := map[string]int{}
	for i, c := range classes {
		classIndex[c] = i
	}

	var images [][]byte
	var labels []byte
	converted := 0
	for _, class := range classes {
		classDir := filepath.Join(src, class)
		err := filepath.Walk(classDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return fmt.Errorf("image-to-idx: traverse %s: %w", path, err)
			}
			if info.IsDir() {
				return nil
			}
			ext := strings.ToLower(filepath.Ext(path))
			if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
				return nil
			}
			f, err := os.Open(path)
			if err != nil {
				return fmt.Errorf("image-to-idx: open %s: %w", path, err)
			}
			img, _, err := image.Decode(f)
			closeErr := f.Close()
			if err != nil {
				return fmt.Errorf("image-to-idx: decode %s: %w", path, err)
			}
			if closeErr != nil {
				return fmt.Errorf("image-to-idx: close %s: %w", path, closeErr)
			}
			images = append(images, data.ScaleImage(img, size, size, 1))
			labels = append(labels, byte(classIndex[class]))
			converted++
			return nil
		})
		if err != nil {
			return err
		}
	}

	if len(images) == 0 {
		return fmt.Errorf("image-to-idx: no images found in %s", src)
	}
	if err := data.WriteIDX(outImages, outLabels, size, size, images, labels); err != nil {
		return fmt.Errorf("image-to-idx: write %s and %s: %w", outImages, outLabels, err)
	}
	if log != nil {
		_, err = fmt.Fprintf(log, "converted=%d classes=%d size=%dx%d\n", converted, len(classes), size, size)
	}
	return err
}
