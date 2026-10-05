// speech-timing-train trains UtauTTS's speech timing Target from an exported cache.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"
)

const phoneNames = "<pad> <unk> sil a i u e o N cl k g s sh z j t ch ts d n h f b p m y r w v ky gy ny hy my ry by py dy ty"
const contextNames = "has_accent accent_position accent_from_end accent_nucleus_position accent_high accent_phrase_start accent_phrase_end word_start word_end breath_position breath_from_end"
const corpus = "Tsukuyomi-chan Corpus Vol.1 (VOICEACTRESS100) + Minnade JSUT Corpus basic5000 BASIC5000_0001-0600, aligned with Montreal Forced Aligner japanese_mfa"

type trainConfig struct {
	Cache, Out, Checkpoint, Resume, Device, ModelID                    string
	Steps, StopAfter, Valid, Batch, Window, EvalEvery, CheckpointEvery int
	Seed                                                               int64
	Context                                                            bool
}

func main() {
	var c trainConfig
	flag.StringVar(&c.Cache, "cache", filepath.Join(os.TempDir(), "gograd-speech-timing", "features.safetensors"), "exported feature cache")
	flag.StringVar(&c.Out, "out", filepath.Join(os.TempDir(), "gograd-speech-timing", fmt.Sprintf("model-%d.safetensors", time.Now().Unix())), "new best inference weights path")
	flag.StringVar(&c.Checkpoint, "checkpoint", "", "resumable training state (default: out + .training.safetensors)")
	flag.StringVar(&c.Resume, "resume", "", "resume from a training checkpoint, not inference weights")
	flag.IntVar(&c.Steps, "steps", 6000, "total planned updates; keep unchanged when resuming")
	flag.IntVar(&c.StopAfter, "stop-after", 0, "stop at this completed update and save training state (0: all planned updates)")
	flag.IntVar(&c.Valid, "valid", 30, "validation utterances")
	flag.Int64Var(&c.Seed, "seed", 0, "model, split and sampler seed")
	flag.BoolVar(&c.Context, "context", false, "use 11 context features")
	flag.StringVar(&c.Device, "device", "auto", "auto, cuda, or cpu")
	flag.StringVar(&c.ModelID, "model-id", "speech-timing-target-v1", "inference checkpoint metadata id")
	flag.IntVar(&c.Batch, "batch-size", 16, "sampled windows per training batch")
	flag.IntVar(&c.Window, "window", 400, "frames per sampled window")
	flag.IntVar(&c.EvalEvery, "eval-every", 250, "validation interval in updates (also first/final update)")
	flag.IntVar(&c.CheckpointEvery, "checkpoint-every", 250, "training checkpoint interval in completed updates")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := trainContext(ctx, c); err != nil {
		fmt.Fprintln(os.Stderr, "speech-timing-train:", err)
		os.Exit(1)
	}
}

func totalFrames(items []utterance) int {
	n := 0
	for _, x := range items {
		n += x.Frames
	}
	return n
}

func checkpointMetadata(id string, context bool, score float64, step int) map[string]string {
	ctx := ""
	if context {
		ctx = contextNames
	}
	return map[string]string{"id": id, "format": "utautts-speech-timing-tcn-1", "phones": phoneNames, "context": ctx,
		"kernel": "5", "dilations": "1 2 4 8 1 2 4 8", "frame_ms": "10.0", "mels": "80",
		"license": "MIT License", "training_corpus": corpus, "license_notices": strings.Join([]string{"licenses/TSUKUYOMI-CORPUS.txt", "licenses/MINNADE-JSUT-CORPUS.txt", "licenses/MFA-Japanese-NOTICE.txt"}, " "),
		"valid_l1": fmt.Sprintf("%.4f", score), "steps": fmt.Sprint(step)}
}
