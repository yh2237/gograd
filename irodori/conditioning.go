package irodori

import (
	"fmt"

	"github.com/yh2237/gograd/autograd"
)

// Conditions contains all checkpoint-projected contexts for one utterance.
type Conditions struct {
	Text        []float32
	TextMask    []bool
	Speaker     []float32
	SpeakerMask []bool
	Caption     []float32
	CaptionMask []bool
}

// Conditioner connects tokenizer, shared ModernBERT, both projectors and the
// reference latent encoder to the denoiser and duration predictor.
type Conditioner struct {
	State     *autograd.SafeTensorFile
	Tokenizer *Tokenizer
}

func (c Conditioner) Encode(text, caption string, reference []float32, maxLength int) (Conditions, error) {
	if c.State == nil || c.Tokenizer == nil {
		return Conditions{}, fmt.Errorf("irodori: missing conditioner checkpoint or tokenizer")
	}
	texts, masks, err := c.Tokenizer.BatchEncode([]string{NormalizeText(text), NormalizeText(caption)}, maxLength)
	if err != nil {
		return Conditions{}, err
	}
	backbone := ModernBERT{State: c.State}
	textHidden, err := backbone.Forward(texts[0], masks[0])
	if err != nil {
		return Conditions{}, err
	}
	textState, err := backbone.Project(textHidden, masks[0], "text")
	if err != nil {
		return Conditions{}, err
	}
	captionHidden, err := backbone.Forward(texts[1], masks[1])
	if err != nil {
		return Conditions{}, err
	}
	captionState, err := backbone.Project(captionHidden, masks[1], "caption")
	if err != nil {
		return Conditions{}, err
	}
	speaker, err := (ReferenceEncoder{State: c.State}).Forward(reference)
	if err != nil {
		return Conditions{}, err
	}
	speakerMask := make([]bool, len(speaker)/768)
	for i := range speakerMask {
		speakerMask[i] = true
	}
	return Conditions{Text: textState, TextMask: masks[0], Speaker: speaker, SpeakerMask: speakerMask, Caption: captionState, CaptionMask: masks[1]}, nil
}

// Denoise predicts one rectified-flow velocity at t for an already encoded
// reference and raw text/caption pair.
func (c Conditioner) Denoise(latent []float32, t float32, conditions Conditions) ([]float32, error) {
	return (CheckpointDiT{State: c.State}).Forward(latent, t, conditions.Text, conditions.TextMask, conditions.Speaker, conditions.SpeakerMask, conditions.Caption, conditions.CaptionMask)
}

// DenoiseCUDA uses CUDA GEMM for the DiT's dense projections while retaining
// the attention and normalization paths on CPU. Hold a CUDAContext through
// this call; each projection is uploaded separately to bound device memory.
func (c Conditioner) DenoiseCUDA(latent []float32, t float32, conditions Conditions) ([]float32, error) {
	return (CheckpointDiT{State: c.State, UseCUDA: true}).Forward(latent, t, conditions.Text, conditions.TextMask, conditions.Speaker, conditions.SpeakerMask, conditions.Caption, conditions.CaptionMask)
}

func (c Conditioner) PredictDuration(conditions Conditions) (float32, error) {
	return (CheckpointDurationPredictor{State: c.State}).Forward(conditions.Text, conditions.TextMask, conditions.Speaker, conditions.Caption, conditions.CaptionMask, true, true)
}

// EncodeReferenceWAVs loads and encodes 48 kHz WAV clips in input order.
func EncodeReferenceWAVs(codec DACVAECodec, paths []string) ([]float32, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("irodori: no reference WAVs")
	}
	var latent []float32
	for _, path := range paths {
		wav, sr, err := ReadWAVMono(path)
		if err != nil {
			return nil, err
		}
		wav, err = ResampleTo48k(wav, sr)
		if err != nil {
			return nil, fmt.Errorf("irodori: %s: %w", path, err)
		}
		piece, err := codec.Encode48k(wav)
		if err != nil {
			return nil, fmt.Errorf("irodori: %s: %w", path, err)
		}
		latent = append(latent, piece...)
	}
	return latent, nil
}
