// irodori-infer currently performs input and checkpoint preflight only.
// It deliberately refuses to write a WAV until the complete reference path
// and SilentCipher watermark have been ported and verified.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/irodori"
)

type paths []string

func (p *paths) String() string     { return strings.Join(*p, ",") }
func (p *paths) Set(s string) error { *p = append(*p, s); return nil }

func run(args []string) error {
	fs := flag.NewFlagSet("irodori-infer", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var text, checkpoint, output string
	var seed int64
	var refs paths
	fs.StringVar(&text, "text", "", "Japanese text to synthesize")
	fs.Var(&refs, "ref", "reference WAV (repeatable)")
	fs.Int64Var(&seed, "seed", 0, "sampling seed")
	fs.StringVar(&checkpoint, "checkpoint", "", "Irodori F32 model.safetensors")
	fs.StringVar(&output, "out", "", "output WAV path (currently disabled)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: irodori-infer -text TEXT -ref WAV -seed N -checkpoint MODEL -out WAV")
		fmt.Fprintln(os.Stderr, "Status: preflight only. Reference conditioning, resampling, DACVAE, RF sampler and the SilentCipher watermark have parity fixtures; production 40-step and full CUDA waveform parity, plus a WAV writer, remain open. No WAV is written.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if text == "" || len(refs) == 0 || checkpoint == "" || output == "" {
		fs.Usage()
		return errors.New("missing required input")
	}
	for _, path := range refs {
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("reference %q: %w", path, err)
		}
	}
	s, err := autograd.OpenSafeTensorFile(checkpoint)
	if err != nil {
		return err
	}
	defer s.Close()
	fmt.Fprintf(os.Stderr, "preflight: normalized_text=%q references=%d seed=%d checkpoint_tensors=%d\n", irodori.NormalizeText(text), len(refs), seed, len(s.Names()))
	return errors.New("inference unavailable: SilentCipher IRDTS watermark is required before writing audio; full CUDA inference remains open; no WAV written")
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
