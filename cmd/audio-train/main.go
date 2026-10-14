// Command audio-train demonstrates Go-only WAV loading and deterministic
// per-sample noise augmentation with a resumable denoising model.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
)

type trainConfig struct {
	WAV, Device, Out, Checkpoint, Resume string
	Steps, StopAfter, Batch, Hidden      int
	Window, ValidCount                   int
	Workers, Prefetch                    int
	SampleRate                           int
	Optimizer                            string
	NoDecaySuffix                        string
	Noise, LearningRate, Momentum        float64
	Seed                                 uint64
	Generate                             bool
	Seconds                              float64
}

func main() {
	var c trainConfig
	flag.StringVar(&c.WAV, "wav", "out/tone.wav", "mono PCM/float WAV path")
	flag.BoolVar(&c.Generate, "generate", false, "write a deterministic synthetic WAV to -wav before training")
	flag.Float64Var(&c.Seconds, "seconds", 8, "generated duration")
	flag.IntVar(&c.SampleRate, "sample-rate", 16000, "generated sample rate")
	flag.StringVar(&c.Device, "device", "cpu", "cpu or cuda")
	flag.StringVar(&c.Out, "out", "out/denoise.safetensors", "inference weights")
	flag.StringVar(&c.Checkpoint, "checkpoint", "", "training state (default: out + .training.safetensors)")
	flag.StringVar(&c.Resume, "resume", "", "resume training state; retain steps/batch/seed/data selection")
	flag.IntVar(&c.Steps, "steps", 400, "total planned updates")
	flag.IntVar(&c.StopAfter, "stop-after", 0, "stop at this completed update and save state (0: all)")
	flag.IntVar(&c.Batch, "batch-size", 64, "windows per update")
	flag.IntVar(&c.Hidden, "hidden", 64, "hidden units")
	flag.IntVar(&c.Window, "window", 64, "samples per window")
	flag.IntVar(&c.ValidCount, "valid-count", 0, "held-out windows (0: one fifth)")
	flag.IntVar(&c.Workers, "workers", 0, "parallel host readers (0: synchronous)")
	flag.IntVar(&c.Prefetch, "prefetch", 0, "pending batches including current (0: two with workers)")
	flag.StringVar(&c.Optimizer, "optimizer", "adamw", "adamw or sgd")
	flag.StringVar(&c.NoDecaySuffix, "no-decay-suffix", "", "optional name suffix whose parameters use no weight decay")
	flag.Float64Var(&c.Noise, "noise", .15, "augmentation noise standard deviation")
	flag.Float64Var(&c.LearningRate, "lr", 0, "OneCycle peak LR (0: .005 AdamW, .02 SGD)")
	flag.Float64Var(&c.Momentum, "momentum", .9, "SGD momentum")
	flag.Uint64Var(&c.Seed, "seed", 7, "signal, augmentation, split and shuffle seed")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if _, err := train(ctx, c, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "audio-train:", err)
		os.Exit(1)
	}
}
