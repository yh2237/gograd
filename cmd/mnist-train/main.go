// Command mnist-train demonstrates Go-only IDX loading, shuffled minibatches,
// evaluation and mid-epoch resume with an MLP and AdamW/OneCycle.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
)

type trainConfig struct {
	DataDir, Device, Out, Checkpoint, Resume        string
	Steps, StopAfter, Batch, TrainLimit, ValidLimit int
	Workers, Prefetch                               int
	Seed                                            uint64
	Download                                        bool
}

func main() {
	var c trainConfig
	flag.StringVar(&c.DataDir, "data-dir", "out/mnist", "raw/gzip MNIST IDX directory")
	flag.BoolVar(&c.Download, "download", false, "download missing MNIST gzip files")
	flag.StringVar(&c.Device, "device", "cpu", "cpu or cuda")
	flag.StringVar(&c.Out, "out", "out/mnist.safetensors", "inference weights")
	flag.StringVar(&c.Checkpoint, "checkpoint", "", "training state (default: out + .training.safetensors)")
	flag.StringVar(&c.Resume, "resume", "", "resume training state; retain steps/batch/seed/data selection")
	flag.IntVar(&c.Steps, "steps", 1000, "total planned updates")
	flag.IntVar(&c.StopAfter, "stop-after", 0, "stop at this completed update and save state (0: all)")
	flag.IntVar(&c.Batch, "batch-size", 64, "examples per update")
	flag.IntVar(&c.Workers, "workers", 0, "parallel host readers (0: synchronous)")
	flag.IntVar(&c.Prefetch, "prefetch", 0, "pending batches including current (0: two with workers)")
	flag.IntVar(&c.TrainLimit, "train-limit", 0, "first N training examples (0: all)")
	flag.IntVar(&c.ValidLimit, "valid-limit", 0, "first N test examples for evaluation (0: all)")
	flag.Uint64Var(&c.Seed, "seed", 7, "model and shuffle seed")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if _, err := train(ctx, c, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "mnist-train:", err)
		os.Exit(1)
	}
}
func outputParent(path string) error {
	if path == "" {
		return nil
	}
	return os.MkdirAll(filepath.Dir(path), 0755)
}
func locateIDX(dir, name string) (string, error) {
	for _, path := range []string{filepath.Join(dir, name), filepath.Join(dir, name+".gz")} {
		info, err := os.Stat(path)
		if err == nil {
			if !info.Mode().IsRegular() {
				return "", fmt.Errorf("%s is not a regular file", path)
			}
			return path, nil
		}
		if !os.IsNotExist(err) {
			return "", err
		}
	}
	return "", fmt.Errorf("missing %s in %s (use -download or supply raw/gzip IDX files): %w", name, dir, os.ErrNotExist)
}
