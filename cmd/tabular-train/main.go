// Command tabular-train demonstrates Go-only numeric CSV loading, deterministic
// feature standardization and mid-epoch resume on a small regression model.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
)

type trainConfig struct {
	Data, Device, Out, Checkpoint, Resume string
	Steps, StopAfter, Batch, Hidden       int
	Rows, Features, ValidCount            int
	Workers, Prefetch                     int
	Optimizer                             string
	LearningRate, Momentum                float64
	Seed                                  uint64
	Generate                              bool
}

func main() {
	var c trainConfig
	flag.StringVar(&c.Data, "data", "out/tabular.csv", "numeric CSV path (header row, last column is the target)")
	flag.BoolVar(&c.Generate, "generate", false, "write a deterministic synthetic dataset to -data before training")
	flag.IntVar(&c.Rows, "rows", 4000, "generated rows")
	flag.IntVar(&c.Features, "features", 4, "generated feature columns")
	flag.StringVar(&c.Device, "device", "cpu", "cpu or cuda")
	flag.StringVar(&c.Out, "out", "out/tabular.safetensors", "inference weights")
	flag.StringVar(&c.Checkpoint, "checkpoint", "", "training state (default: out + .training.safetensors)")
	flag.StringVar(&c.Resume, "resume", "", "resume training state; retain steps/batch/seed/data selection")
	flag.IntVar(&c.Steps, "steps", 400, "total planned updates")
	flag.IntVar(&c.StopAfter, "stop-after", 0, "stop at this completed update and save state (0: all)")
	flag.IntVar(&c.Batch, "batch-size", 64, "rows per update")
	flag.IntVar(&c.Hidden, "hidden", 32, "hidden units")
	flag.IntVar(&c.ValidCount, "valid-count", 0, "held-out rows (0: one fifth)")
	flag.IntVar(&c.Workers, "workers", 0, "parallel host readers (0: synchronous)")
	flag.IntVar(&c.Prefetch, "prefetch", 0, "pending batches including current (0: two with workers)")
	flag.StringVar(&c.Optimizer, "optimizer", "adamw", "adamw or sgd")
	flag.Float64Var(&c.LearningRate, "lr", 0, "OneCycle peak LR (0: .01 AdamW, .05 SGD)")
	flag.Float64Var(&c.Momentum, "momentum", .9, "SGD momentum")
	flag.Uint64Var(&c.Seed, "seed", 7, "model, split and shuffle seed")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if _, err := train(ctx, c, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "tabular-train:", err)
		os.Exit(1)
	}
}
