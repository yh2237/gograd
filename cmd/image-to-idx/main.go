package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
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

	entries, err := os.ReadDir(*src)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var classes []string
	for _, e := range entries {
		if e.IsDir() {
			classes = append(classes, e.Name())
		}
	}
	sort.Strings(classes)
	if len(classes) == 0 {
		fmt.Fprintln(os.Stderr, "image-to-idx: no class subdirectories found")
		os.Exit(1)
	}
	classIndex := map[string]int{}
	for i, c := range classes {
		classIndex[c] = i
	}

	var images [][]byte
	var labels []byte
	converted := 0
	for _, class := range classes {
		classDir := filepath.Join(*src, class)
		err := filepath.Walk(classDir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info == nil || info.IsDir() {
				return nil //nolint:errcheck
			}
			ext := strings.ToLower(filepath.Ext(path))
			if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
				return nil
			}
			f, err := os.Open(path)
			if err != nil {
				return nil
			}
			img, _, err := image.Decode(f)
			f.Close()
			if err != nil {
				return nil
			}
			pixels := data.ScaleImage(img, *size, *size, 1)
			quantized := make([]byte, len(pixels))
			for i, v := range pixels {
				quantized[i] = byte(math.Round(float64(v) * 255))
			}
			images = append(images, quantized)
			labels = append(labels, byte(classIndex[class]))
			converted++
			return nil
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	if len(images) == 0 {
		fmt.Fprintln(os.Stderr, "image-to-idx: no readable images found")
		os.Exit(1)
	}
	if err := data.WriteIDX(*outImages, *outLabels, *size, *size, images, labels); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("converted=%d classes=%d size=%dx%d\n", converted, len(classes), *size, *size)
}
