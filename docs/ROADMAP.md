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
| Text | Cached ModernBERT-ja Unigram tokenizer matches 65 reference cases, including BOS, padding and truncation; streamed 25-layer ModernBERT and both text/caption projectors match real-checkpoint hidden-state slices | Encoder training/backward and optimized CUDA execution |
| DiT/duration | Integrated raw text/caption, eight-clip speaker context, real-checkpoint denoiser and duration parity; 1.463 GB DiT host cache reduces repeated forward time | DiT training/backward and complete CUDA execution |
| RF | Injected PyTorch seed noise, sway/linear grid, independent/joint/alternating CFG and Euler loop; three-step real-checkpoint final-latent fixture | Native PyTorch CPU/CUDA RNG reproduction, longer/full-sentence sampler parity, optional score rescale and speaker KV scaling |
| Reference/audio | WAV and K-weighted loudness normalization, torchaudio-compatible 44.1-to-48 kHz resampling, deterministic DACVAE encode, reference latent encoder, safe `.pth` conversion and deterministic decoder parity | More input rate/channel fixtures and faster codec kernels |
| Output | One-step 2.32-second sentence reaches waveform sample parity internally | SilentCipher `IRDTS` watermark and production 40-step/full CUDA parity; CLI refuses WAV output |

For a 256x1024 RMSNorm microbenchmark this round, standalone Go CPU took
419,073 ns/op, reusable graph Go CPU took 3,170,007 ns/op, and PyTorch CPU
(one thread) ranged from 195,450 to 272,600 ns/op in two runs. With a
synchronized host readback per call, reusable graph Go CUDA took 849,393 ns/op
and PyTorch CUDA took 245,700 ns/op. These separate runs are sensitive to
shared GPU load and do not measure full-model inference. The graph composition
needs a fused kernel before it is a competitive RMSNorm implementation.

## Checkpoint-backed inference progress (2026-10-08)

`irodori.ModernBERT` streams the 25-layer F32 backbone; `Project` applies the
checkpoint's text/caption residual MLP projectors and final RMSNorm. On three
fixture lengths (16, 32, 160), final-hidden-state slice max absolute errors
were 3.34e-6, 1.79e-6, and 2.98e-6. Text/caption projector slice errors
stayed below 2.87e-6. The 160-token case exercises local attention.

`CheckpointDurationPredictor` matches two real-checkpoint cases to at most
2.38e-7 absolute error. `CheckpointDiT` matches all 128 values of one
12-block denoiser forward to 1.16e-5 maximum absolute error. These stage-three
fixtures supply fixed already-encoded text, speaker and caption states. They
verify the full duration and DiT modules but do not verify the reference
latent encoder or an integrated prompt-to-latent call.

Reference fixture timings (CPU, PyTorch, six threads) were 74.7/85.2/304.2 ms
for ModernBERT at 16/32/160 tokens, 211.5 ms cold and 4.68 ms warm for the
duration cases, and 336.2 ms for the DiT forward. Go CPU streamed weights on
each call: 1549/1721/2908 ms for the encoder and projectors, 195/163 ms for
duration, and 4172 ms for DiT. These were not paired steady-state benchmarks;
Go pays file reads on every call, and the first PyTorch duration call includes
warmup. Cache weights and add fused CPU/CUDA kernels before treating the ratios
as representative inference performance.

## Reference audio and integrated inference progress (2026-10-08)

The read-only cached DACVAE `weights.pth` can be converted with
`tools/convert_irodori_codec.py SOURCE.pth DEST.safetensors`, where the
destination **must be outside this repository**. The converter uses
`torch.load` only on the trusted local checkpoint, folds weight normalization
into F32 convolution weights, and writes safetensors with the model kwargs in
metadata. No checkpoint weights or converted weights belong in test fixtures.
`IRODORI_CODEC_SAFE` selects that local converted file for Go parity tests.

The eight specified 48 kHz PCM reference WAVs pass Go loading and K-weighted
loudness normalization parity. A 44.1-to-48 kHz real-clip excerpt matches
torchaudio's default sinc resampler to 2.98e-8 max absolute error. The AudioTools/Julius loudness window includes
a zero-padded trailing block, which matters for short clips. A four-frame
DACVAE decoder waveform segment matches PyTorch to 1.13e-6 max absolute
error (Go 968 ms, PyTorch 678 ms). The speaker encoder on the eight-clip
PyTorch latent fixture matches sampled speaker-state values to 2.98e-6
(Go 1.58 s, PyTorch 166 ms). End-to-end loading and encoding of all eight
WAVs matches every stored latent element to 6.18e-5 maximum absolute error;
the sampled final speaker state differs by at most 4.41e-6. The Go codec and
speaker passes took 93.2 s and 1.50 s in that run.

Raw Japanese text and caption plus the eight-clip latent fixture feed the
shared ModernBERT, projectors, speaker encoder, 12-block DiT and duration
predictor in one Go call. Sampled denoiser velocity differs from PyTorch by
7.36e-6 and duration matches at reported F32 precision. A three-step sway
Euler sample with independent text/speaker/caption CFG and injected PyTorch
seed noise differs by 0.00107 on sampled final-latent elements. The same
real-checkpoint three-step fixture reaches 1.97e-6 for joint CFG and
1.24e-5 for alternating CFG. The larger independent-CFG difference still
needs a branch-by-branch audit of the reference's batched evaluation. The RNG is
still injected from a small fixture; Go does not yet reproduce PyTorch's
generator bit-for-bit. This is a short numerical parity case, not a
full-sentence audio benchmark.

`PreloadDenoiser` retains 1,462,937,728 bytes of F32 DiT weights in host RAM
for repeated sampling calls. In one four-frame CPU comparison, streamed and
resident calls took 3,428 ms and 2,520 ms, with identical output; preloading
took 1,401 ms. `LinearRowsCUDA` exercises the existing CUDA matmul kernel
with per-layer uploads, matching one real-checkpoint projection to 1.19e-7.
For that 4x32-to-1280 projection, 20 repeated Go CUDA calls averaged
0.209 ms each with upload/readback; PyTorch CUDA averaged 0.064 ms with
resident input/weight and synchronized output. The transfer difference makes
this a path smoke test, not a comparable model-level throughput claim.
One 2.32-second sentence with all eight raw reference WAVs, raw text/caption,
predicted 116 latent steps, one independent-CFG Euler step and DACVAE decode
matches PyTorch final-latent slices to 3.40e-5 and 256 sampled waveform
points to 4.43e-5 max absolute error. Go CPU stages took 92.6 s for eight
codec encodes, 4.42 s conditioning, 0.89 s DiT preloading, 99.4 s sampling
and 14.3 s decoding (212 s test total). PyTorch CPU took 14.4 s, 1.33 s and
2.95 s for codec, sampler and decoder, respectively; its conditioning cost
was not separately measured in that full run. These are single-run timings.
The F32 DiT/ModernBERT/speaker checkpoint occupies about 3.06 GB before
intermediates, so an all-CUDA run would exceed the shared ~3 GB budget.
End-to-end CUDA execution and a 40-step full sentence comparison remain open.
The same full sentence through the mixed backend (CPU codec, text encoder,
attention and norms; CUDA DiT dense projections) had 4.03e-5 final-latent
slice error and 4.50e-5 sampled waveform error. Its Go sampler took 94.2 s;
the other stages took 90.7 s for codec, 4.19 s for conditioning, 0.91 s
for preloading and 14.6 s for decode (205 s test total). A corresponding
PyTorch F32 CUDA DiT-only one-step batched CFG call took 2.79 s and peaked
at 1,531,809,280 allocated GPU bytes; its first latent slice differed from
the CPU reference fixture by 1.14e-5. The Go path transfers each projection
separately and evaluates CFG branches serially, so the GPU matmul does not
yet offset CPU attention and transfer costs. This comparison isolates the
denoiser stage and is not an end-to-end PyTorch CUDA benchmark.

**Release gate:** the Go CLI continues to refuse WAV output. The reference
pipeline applies a SilentCipher `IRDTS` watermark to generated audio. Port
and verify it before exposing any output WAV. More input audio formats and
production 40-step waveform parity remain open as well.
