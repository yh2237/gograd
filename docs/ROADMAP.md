# PyTorch replacement roadmap for UtauTTS

This inventory was checked against the read-only `D:/project/UtauTTS` tree on
2026-10-08. `out/` contains many dated experiments; the rows distinguish a
current production path from teacher-data and evaluation tools. A Go runtime
must load the original weights and reproduce outputs before replacing Python.

| Consumer in UtauTTS | Current Python/PyTorch use | Missing gograd capability |
| --- | --- | --- |
| `out/reference-tts-20261008b/Irodori-TTS`, `out/irodori-teacher-20261008` | Irodori teacher WAVs from a ModernBERT-ja text/caption encoder, reference DACVAE encoder, duration predictor, DiT/RF sampler, DACVAE decoder, SilentCipher watermark | HF tokenizer JSON, ModernBERT sliding attention/RoPE, gated multi-context attention, Low-Rank AdaLN, `weights.pth` conversion, transposed convolution/codec, exact reference audio preprocessing, seeded sampler, watermark |
| `out/irodori-teacher-20261008/pipeline.py`, `out/mfa-align-20261002`, `out/duration-mfa-20261002` | Montreal Forced Aligner subprocess for phone intervals; Python/SciPy audio conversion and PyTorch duration experiments | A phone alignment engine and acoustic model are separate from generic tensor work; add WAV/resampling and alignment data interchange first. Existing Go speech-timing trainer already covers one related TCN. |
| `out/asr-eval-20261004/asr_eval.py` | Transformers pipeline with `openai/whisper-large-v3`, FP16 CUDA, for evaluation | STFT/log-mel frontend, Whisper encoder/decoder, tokenization, KV cache, beam/greedy decoding, FP16 storage and kernels |
| `out/reference-tts-20261007/CosyVoice`, `out/cosy-duration-20261008`, `out/cosy-memorize-20261008` | CosyVoice3 prompt/reference speech experiments and duration/prosody comparisons; upstream code includes LLM, flow matching, DiT, HiFiGAN | LLM cache/sampling, attention/transformers, flow solver, conv-transpose, vocoder, mel/audio frontends, model-format conversion |
| `out/reference-tts-20261007/generate.py`, `generation-qwen.json` | Qwen3-TTS comparison generation | Qwen-style transformer, RoPE, KV cache, codec/audio tokenizer, generation loop and weight loader; audit the exact variant before implementation |
| Other dated `out/*/train*.py`, `apply.py`, `warp.py` | Prototypes and training/evaluation scripts using PyTorch tensors, optimizers and `.pt` snapshots | Many can use existing `autograd` ops and AdamW; migrate only still-used experiments after their data and checkpoint formats are identified. |
| `tools/openjtalk-feature-bridge.py` and `tools/openjtalk_features.py` | Python feature extraction/bridge around OpenJTalk, not mainly PyTorch | Preserve the current external OpenJTalk boundary or build a separate Japanese front-end; it is outside the tensor engine. |

## Priority and gates

1. **Shared numerical blocks.** Keep RMSNorm, RoPE (including partial-head
   rotation and configurable theta), SwiGLU, attention masks, grouped query
   attention, and Low-Rank AdaLN as reusable `autograd` operations or modules.
   Add CPU/CUDA forward and gradient fixtures for each shape family. This round
   adds graph-composed `RMSNormLast` with CPU/CUDA forward parity; its composed
   kernels are a correctness baseline, not the final fused implementation.
2. **Irodori text and model.** Finish tokenizer JSON, exact normalization,
   ModernBERT-ja and shared text/caption projector, reference latent encoder,
   duration predictor, 12 DiT blocks, and all CFG modes. Verify each layer from
   small PyTorch fixtures before attempting a full checkpoint. Stream the 714
   F32 safetensors tensors with bounded live memory. The existing streaming
   reader proves one real tensor, but there is no runnable state dictionary.
3. **Audio and release gate.** Implement deterministic DACVAE `.pth` conversion
   to a documented safe format, encode/decode, 48 kHz WAV preprocessing,
   resampling, loudness normalization, tail handling and SilentCipher `IRDTS`
   watermark. Compare latent and waveform segments against PyTorch. **Do not
   present generated audio as a drop-in Irodori replacement or publish an
   inference WAV command until the watermark and full waveform parity pass.**
4. **Performance.** Add FP16 and full BF16 storage, mixed-precision rules,
   memory planning, fused RMSNorm/AdaLN/attention and CUDA graph plans under
   a 3 GB model-specific budget. Benchmark identical weights, prompts,
   references, seeds, output lengths and precision on CPU and CUDA. A primitive
   microbenchmark cannot stand in for end-to-end inference.
5. **Other consumers.** Reuse the transformer/cache/audio stack for Whisper
   evaluation and CosyVoice/Qwen studies. Keep MFA alignment as a separate
   research track with phone-boundary quality tests; port stable, still-used
   PyTorch training prototypes after Irodori is gated.

## Current Irodori stage status

| Stage | Verified in Go | Open gate |
| --- | --- | --- |
| Text | Ordered normalization has a 12-case fixture from the reference; duration features have PyTorch fixtures | HF fast tokenizer `tokenizer.json` token IDs, padding/truncation, full text/caption encoder |
| DiT/duration | Standalone RMSNorm, RoPE, timestep, Low-Rank AdaLN and duration features have CPU fixtures; reusable graph RMSNorm has CPU/CUDA forward parity | Full modules, intermediate layer parity, complete checkpoint mapping |
| RF | Linear/sway grid and temporal score rescale fixtures | Seeded PyTorch noise, CFG branch combinations and Euler loop |
| Reference/audio | No codec inference yet | WAV normalization/resample, latent encoder, DACVAE encode/decode and `.pth` conversion |
| Output | No Go waveform produced | SilentCipher watermark and end-to-end waveform parity |

For a 256x1024 RMSNorm microbenchmark this round, standalone Go CPU took
419,073 ns/op, reusable graph Go CPU took 3,170,007 ns/op, and PyTorch CPU
(one thread) ranged from 195,450 to 272,600 ns/op in two runs. With a
synchronized host readback per call, reusable graph Go CUDA took 849,393 ns/op
and PyTorch CUDA took 245,700 ns/op. These separate runs are sensitive to
shared GPU load and do not measure full-model inference. The graph composition
needs a fused kernel before it is a competitive RMSNorm implementation.
