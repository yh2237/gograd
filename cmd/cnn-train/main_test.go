package main

import (
	"io"
	"path/filepath"
	"testing"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

// A complete training exercise, independent of the operation fixtures: learn
// spatial patterns, save the trained model, reconstruct it and verify logits.
func TestCNNTrainingAndReload(t *testing.T) {
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(string(device), func(t *testing.T) {
			if device == tensor.CUDA && !cuda.Available() {
				t.Skip("CUDA unavailable")
			}
			file := filepath.Join(t.TempDir(), "classifier.safetensors")
			stats, err := train(device, 40, file, io.Discard)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("initial=%g final=%g accuracy=%g reload_max_abs=%g", stats.initialLoss, stats.finalLoss, stats.accuracy, stats.reloadMaxAbs)
			if stats.finalLoss >= stats.initialLoss*.2 || stats.accuracy < .95 {
				t.Fatal("CNN did not learn spatial classification")
			}
			if stats.reloadMaxAbs > 1e-5 {
				t.Fatal("checkpoint reload changed inference")
			}
		})
	}
}
