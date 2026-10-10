package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const mnistURL = "https://storage.googleapis.com/cvdf-datasets/mnist"

var mnistFiles = []string{"train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"}

func downloadMNIST(ctx context.Context, dir, baseURL string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	client := &http.Client{Timeout: 2 * time.Minute}
	for _, name := range mnistFiles {
		if _, err := locateIDX(dir, name); err == nil {
			continue
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err := downloadFile(ctx, client, baseURL+"/"+name+".gz", filepath.Join(dir, name+".gz")); err != nil {
			return err
		}
	}
	return nil
}
func downloadFile(ctx context.Context, client *http.Client, url, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	response, err := client.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, response.StatusCode)
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".mnist-download-*")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if _, err := io.Copy(f, response.Body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(temp, path)
}
