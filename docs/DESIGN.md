# General tensor and autograd design

## Contract and current slice

`autograd` is a define-by-run float32 graph. A tensor has a logical shape,
strides, a storage offset, a device, data or a CUDA buffer, an optional
accumulated gradient, and a backward closure. Reshape, transpose, permute,
slice, and expand create shared-storage views; `Contiguous` materializes one
when a kernel requires row-major data. CUDA graph values,
gradients, optimizer moments and clipped gradients remain in device buffers.
Training uploads inputs and reads the scalar loss. Parity tests and explicit
state export also read tensors back for inspection.

The public design uses `Tensor{Shape, Strides, DType, Device}` with CPU slices
or CUDA buffers. `Backward` returns CUDA intermediates to the allocation
pool; `ReleaseGraph` releases CPU intermediates for reuse and also frees
unused CUDA inference graphs. Tensor/view allocations now have shared storage
leases, and graph edges have independent dependency counts. A Close request is
deferred internally while a child still needs a closed ancestor's values;
ReleaseGraph defers reclaiming shared nodes until their last dependent is done.
RetainData/RetainGrad can keep a view's allocation without keeping its base
history. Chained views may still materialize their parent. Dtypes start with float32,
then add float16 and bfloat16 compute with float32 accumulation and int64
indices. Operators must reject mixed devices and unsupported dtype pairs.
Operators accept non-contiguous inputs by materializing them before kernel
dispatch. Views keep their base storage alive.

## Graph and operators

Each op records its parents and a vector-Jacobian product when recording is
enabled. Backward topologically traverses the graph in reverse, accumulating
all contributions to leaves. Intermediate gradients are discarded unless
`RetainGrad` is called. `Detach` returns data without history and preserves
execution context identity. `ExecutionContext` isolates recording for bound
tensors; `New`/`Zeros` on the context bind inputs, and `BindModule` binds every
registered parameter/buffer before a forward. Operators and shared-storage
views inherit the context, and reject combinations of distinct explicit
contexts. NoGrad uses an atomic per-context scope depth, so nesting and panics
restore correctly and separate contexts do not affect one another.

ExecutionOptions adds context-local BF16 precision and attention selection.
A context's zero value means FP32/auto, independent of the compatibility
globals. Options are copied under a per-context mutex; SetOptions validates
before changing state. Autocast temporarily changes precision and restores it
after nesting/panics. CUDA GEMM nodes capture the option value in their backward
closures, keeping operand preparation and backward dispatch consistent after
a scope exits or policy changes. Attention selects its implementation in
forward and retains that implementation's derivative closure. CUDA capture
records the chosen kernels; replay does not resolve Go-side options again.
Scopes affect the whole context, and a multi-op model forward is not an atomic
policy transaction: configure between forwards and use separate contexts for
independent workers.

The package-level API retains a shared default scope for unbound tensors.
Fully isolated execution needs bound parameters too: a parameter-only
subexpression such as an embedding or weight view must not enter the default
scope before meeting a bound input. `cmd/cnn-train` exercises this migration.
Inference results keep their lifetime parents for reclamation, but have no
derivative callbacks. The context does not yet own storage or CUDA streams and
does not synchronize mutation of shared tensors/optimizers. Storage versions
now track CopyFrom and eager optimizer writes across aliases; backward checks
operand/output snapshots before changing gradients. Raw CPU slice or CUDA
buffer writes/captured replay do not update those host-side versions.

Default backward consumes saved history. BackwardWithOptions can retain it for
another loss/pass; intermediate adjoints are fresh each time and leaf gradients
accumulate. Saved im2col, normalization statistics, masks/targets and index
buffers outlive their source handles as needed. CUDA nodes being visited by
backward are protected from disposal until their callbacks finish, so completed
children can drop edges without reclaiming an ancestor still awaiting backward.
Data retention and history retention are separate. Explicit graph release is
still required for inference and retained histories; abandoned graphs are not
automatically managed by Go GC.

The operation registry names the CPU and CUDA implementation of every public
differentiable operator and selects its backend. An AST test inventories these
operators and requires a registry entry and a dispatch call in each public
entry point. A mutation check confirmed the test fails when a dispatch is
removed. The current table selects by operation and device; callable kernels
keyed by dtype and layout remain future work. CPU uses blocked SGEMM with a
runtime-selected AVX2/FMA 8x8 microkernel (4x8 with `GOGRAD_SGEMM_8X8=0`)
and pure Go fallback; CPU elementwise
ops and reductions use multiple cores, and size pools reuse buffers. CUDA
Conv1d uses im2col/col2im plus cuBLAS SGEMM. Regular batched matmul and its
gradients use cuBLAS strided-batched GEMM; broadcast weight gradients flatten
the batch into one GEMM. Irregular batches retain a loop fallback. NVRTC kernels implement
elementwise broadcasting, shape transforms, embedding, reductions,
normalization, masked losses, softmax, cross-entropy, dropout, and clipping.
CUDA N-D indexing supports ranks through six for broadcasting, reduction,
and views. The CUDA driver context is tied to
an OS thread, so callers hold `NewCUDAContext` through the training step.
The graph currently uses one stream. Captured graphs reserve their temporary
pool allocations until `Graph.Close`, and cuBLAS is bound to the capture
stream. This does not yet support overlapping independent streams or concurrent
captures. The CUDA pool now rounds requests into size classes (256 B, 4 KiB,
64 KiB or 1 MiB alignment by size), caches at most 2 GiB by default, and
evicts least-recently-used cached buffers above that limit. `MemoryStats`
reports active, cached and driver-reserved bytes and their peaks;
`SetPoolCacheLimit` can lower the cap, and `ReleasePool` frees cached memory.
Graph-reserved pointers are removed from the ordinary cache until graph close.
The pool is still global rather than keyed by device and stream.

Warp reductions fuse softmax forward and backward. The materialized attention path
uses batched QK and PV GEMMs around fused masked scale/softmax forward and
backward kernels. Layer norm has direct fused row kernels, and the encoder's
feed-forward path fuses bias with GELU and residual addition on CUDA when
dropout is zero. Optional tiled attention uses online softmax, stores only the
output and row statistics, and recomputes scores in backward. It supports an
additive mask and avoids the quadratic score buffer. The current CUDA tiles
trade substantial time for memory; automatic selection uses the tiled path
only when estimated score scratch reaches 1 GiB. BF16 autocast caches GEMM
operands and carries BF16 shadows through eligible views and fused pointwise
kernels. GEMMEx accumulates into FP32; weights and optimizer moments remain
FP32. This mode and attention selection belong to ExecutionOptions for bound
tensors; only the compatibility API uses process-wide switches. CPU, Conv2d
and tiled flash attention currently stay FP32. Full BF16 tensor storage and
dtype-aware dispatch remain future work.

Broadcasting aligns dimensions from the right. Each pair must match or one
dimension must be 1. Backward sums over expanded dimensions. Matmul applies
this to batch dimensions, with the last two dimensions as matrices. Reduction
axes are explicit and removed from output shape; scalar output has shape
`[]`. Concat and slice use zero-based axes and half-open slice bounds.
Embedding backward scatter-adds repeated indices and can skip a padding row.
`IndexBuffer` uploads fixed IDs once for captured training steps. Conv1d uses
`[batch,time,channels]`, weights `[out,in,kernel]`, odd kernel, cross
correlation, dilation and symmetric zero same-padding. GroupNorm computes
statistics over time and channels per sample and group. LayerNorm computes
statistics over the final dimension. Masked loss uses non-NaN target frames
and divides by all valid frame/channel elements, matching the Acoustic
reference. A zero-valid-element loss currently returns zero. Softmax and
log-softmax are stable along any axis, and cross-entropy averages over targets
other than the ignore index. Attention computes scaled QKᵀ, adds an optional
broadcastable mask, applies softmax, and multiplies by V. `MultiheadAttention`
uses PyTorch's packed projection layout; `TransformerEncoderLayer` is a
batch-first pre-norm encoder with GELU and an adjustable feed-forward width.
Dropout hashes `(seed, element index)` so CPU and CUDA reuse the same mask
for backward; its random stream deliberately differs from PyTorch's.

## Modules and training

`Acoustic` is the first graph-composed model. Its parameter names match the
PyTorch reference, and the small fixture tests all of their gradients. The
`Module` registers ordered named parameters, buffers and children and
provides `StateDict`, `LoadStateDict`, `SaveSafeTensors`, and
`LoadSafeTensors`. The F32 safetensors loader validates names, shapes,
offsets, and byte lengths before changing weights. Train/eval propagation,
device/dtype moves, atomic checkpoint writes, and optimizer state serialization
remain.
Reusable Embedding, Conv1d, LayerNorm, Linear, Dropout and GELU layers now
compose `SpeechTiming` through a plain struct and child `Module` nodes.
Its state names (`phone.weight`, `inp.weight`, `blocks.0.weight`,
`norms.0.bias`, `out.weight`, and peers) and convolution weight layout match
the UtauTTS PyTorch trainer and Go `speechtiming.LoadTCN`. `Train(false)`
disables dropout for validation. Safetensors metadata carries UtauTTS's
format, phones, context, kernel, dilations, mel count, license and best
validation score. The separate feature-cache exporter writes I64 phone IDs
and F32 continuous inputs and targets to safetensors; it records the Python
validation shuffle for a comparable split. This is a task-specific cache
bridge, not yet a general data-loader API.

Parameters are graph leaves; nontrainable running statistics are buffers.
Names are stable dotted paths and duplicate aliases are stored once.

AdamW uses decoupled weight decay, float32 moments and bias correction.
Global norm clipping happens before optimizer step. The OneCycle schedule
uses PyTorch's two phase cosine default: initial LR is max/25, final LR is
initial/10000, and `pct_start` sets the warmup boundary. Future scheduler
state must include step count. Data loading should separate dataset indexing,
sampling, collation, pinned host buffers, worker prefetch and asynchronous
device transfer. Batch seeds and sharding must be reproducible.

Serialization uses safetensors-compatible tensor payloads and metadata:
an 8-byte little-endian header length, JSON descriptors with dtype, shape and
byte offsets, followed by contiguous payload. Put model architecture and
optimizer/scheduler metadata in a separate versioned manifest. Validate
shapes, ranges, duplicate names and size limits before allocation; write
atomically. The current reader checks shapes and ranges but does not yet
enforce a file-size cap, reject overlapping payloads, or write atomically.
Existing `nn` JSON checkpoints remain readable during migration.

## General Conv2d

`autograd.Conv2d` adds NCHW cross-correlation with PyTorch-compatible
`[out,in/groups,kH,kW]` weights. Stride, symmetric padding and dilation are
independent height/width pairs; groups includes depthwise output multipliers.
The bias is optional. `Conv2dLayer` registers named parameters through the
existing `Module` and safetensors APIs.

Both backends lower one batch/group into depth-major spatial tiles and run
GEMM. A tile contains at most 512 output positions and targets a 4 MiB budget
for the im2col/output pair. Backward gathers an upstream tile, recomputes
im2col for the weight gradient, and scatters the input gradient from
`weight-transpose * upstream`. CPU accumulates per-tile weight contributions;
CUDA adds them directly into the leaf gradient with cuBLAS beta=1 and uses
atomic col2im scatter for input gradients. No full lowered matrix is retained
by the graph, and inference releases all temporary device buffers before
returning its output. The path is FP32, independent of optional BF16 shadows.

The operation is registered and covered by the source dispatch inventory.
Committed checkpoint-free PyTorch fixtures exercise batched, grouped,
depthwise, rectangular, even and strided/dilated cases on CPU/CUDA; finite
differences independently check gradients. The `cmd/cnn-train` example verifies
that two convolution blocks compose with reduction, classification loss,
AdamW and checkpoint reload to learn a spatial classification task in pure Go.

## NCHW pooling

MaxPool2d, AvgPool2d and AdaptiveAvgPool2d share NCHW window geometry and
separate CPU/CUDA implementations registered at their public entry points.
Geometry is validated before materializing views or allocating outputs. Max
pooling supports dilation and saves one selected index per output only for
recorded graphs. Those indices belong to the derivative history, so retained
backward and closing an ancestor cannot invalidate them prematurely. Average
and adaptive-average derivatives need only the captured geometry/divisor.

CPU forward partitions output positions. Backward partitions whole N*C planes
using output size to choose worker count: overlapping windows accumulate
serially within each plane, avoiding races on input gradients. CUDA runs one
thread per output window and uses atomic scatter-add for backward. All values
remain FP32; device kernels operate without host staging and can be captured
and replayed. NoGrad max pooling avoids the saved-index allocation.

Ceil mode removes windows starting entirely in right/bottom padding. Average
divisors distinguish explicit zero-padding from positions beyond that padding
in a partial ceil window. Adaptive bins use floor/ceil boundaries and may
overlap when input size is not divisible by output size. Fixtures compare all
outputs and VJPs with PyTorch; independent finite differences, alias/retained
history tests and CUDA capture exercise composition. The CNN example now
includes max downsampling and adaptive global averaging before its classifier.

## Batch normalization and mutable state

BatchNorm normalizes all non-channel dimensions of rank 2..6 tensors. A small
outer/channel/inner descriptor supports both channels-first and channels-last
layouts after view materialization. The functional operation accepts optional
affine parameters and paired running buffers. BatchNormLayer registers weight,
bias, running mean/variance and a batch counter through Module, and its mode
callback participates in recursive Train propagation. NoGrad does not disable
training-state updates.

Training computes biased variance for normalization and uses the unbiased
estimate for running variance. CPU and CUDA moment reductions accumulate in
float64; each forward saves FP32 mean/inverse-standard-deviation values in an
independent ephemeral Tensor. That Tensor is a saved lifetime/context dependency;
running buffers themselves are not backward dependencies. Consequently, a later
forward can update running state without invalidating an earlier graph, and
evaluation backward survives closing the running buffers. Input/affine versions
are still validated, and retained history keeps its saved statistics alive.

CUDA uses one reduction block per channel plus elementwise forward/input-VJP
kernels. Weight/bias VJPs reduce per channel without atomic scatter. A separate
counter kernel precedes statistics updates, so cumulative momentum reads the
new device-side count without racing another channel's block. Capture/replay
updates state on device, including the counter; no per-forward readback is
required. The current Module state remains F32, so num_batches_tracked is an
F32 scalar rather than I64 and integer precision is bounded by 2^24.

Checkpoint reload tests reconstruct layer configuration and explicitly enter
evaluation. The CNN example now trains Conv2d/BatchNorm/ReLU blocks, saves affine
and running state, and verifies logits after reload on CPU/CUDA. PyTorch fixtures,
numerical gradients, independent-worker race tests, history/lifetime tests and
captured-vs-eager state comparisons cover the general operation and layer.

## Migration and phases

1. Extend the operation registry from backend selection to callable kernels
   keyed by dtype and layout, and make views composable without intermediate
   materialization. Tensor-bound execution contexts now isolate recording,
   precision and attention selection, with shared-storage lifetime management;
   extend isolation to CUDA streams/devices and allocation pools. Port
   `nn.Linear` and `nn.Conv1d` to graph operators, then make their explicit
   backward methods compatibility wrappers.
2. Move `gputcn` layers and its sequence loss onto the same operators. Retain
   its optimized CUDA graph path as a compiled execution plan. Add general
   prefetching dataset loaders and continue trainer adoption. Resumable AdamW /
   OneCycle / model checkpoints, restorable index/window samplers, and recursive
   `Module.Train` propagation are implemented in v1.2.0. `speech-timing-train`
   additionally retains the data split and best model for complete resume.
3. Improve tiled attention occupancy and backward throughput, complete BF16
   storage and dtype-aware kernels, and add FP16 loss scaling. Extend the
   general CUDA graph capture API beyond fixed Acoustic and transformer
   batches. Extend CPU SIMD beyond SGEMM while retaining pure Go fallbacks.
   Validate against PyTorch on each dtype and shape family before removing
   legacy paths.

## Measured status (2026-10-04)

Median full training-step times from three paired gograd/PyTorch runs in one
session, with 12 CPU threads. The transformer has four encoder layers,
model width 256, four heads, feed-forward width 1024, batch 16 and sequence
256. Acoustic has hidden width 384, batch 24 and 400 frames. BF16 uses PyTorch
autocast and gograd's FP32-authoritative BF16 shadow path. Units are ms.

| Model and runtime | CPU FP32 | CUDA FP32 eager | CUDA FP32 graph | CUDA BF16 eager | CUDA BF16 graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acoustic PyTorch | 2505.9 | 58.5 | 53.7 | 47.1 | 42.6 |
| Acoustic gograd | 3876.4 | 82.3 | 82.0 | 63.8 | 64.1 |
| Transformer PyTorch | 1291.6 | 25.9 | 23.9 | 13.5 | 12.0 |
| Transformer gograd | 2276.1 | 27.2 | 27.8 | 22.4 | 23.2 |

The transformer CPU ratio is 1.76x, meeting the 2.5x target. Acoustic CPU is
1.55x, narrowly above its 1.5x target. Graph capture saves little on gograd
for these GEMM-heavy workloads. Eager BF16 improves both models over gograd
FP32 but still trails PyTorch BF16, especially on the transformer (1.66x).
The default 8x8 SGEMM kernel won paired CPU trials against the 4x8 fallback:
Acoustic medians were 3783.4 versus 4157.8 ms and transformer medians were
2085.0 versus 2242.9 ms. CPU profile samples remain dominated by SGEMM
(about 61% on Acoustic and 33% on transformer after the changes), followed by
indexing and pointwise gradients. Profiles and benchmark commands are described
in README.

For batch 1, four heads and head width 64, separate CUDA-event attention
forward/backward medians and peak extra live buffer use were:

| Sequence | Materialized step | Tiled step | Materialized peak | Tiled peak |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.366 ms | 0.766 ms | 3.50 MiB | 1.01 MiB |
| 1024 | 1.601 ms | 9.606 ms | 50.00 MiB | 4.03 MiB |
| 4096 | 18.916 ms | 127.889 ms | 776.00 MiB | 16.12 MiB |

The tiled path passed forward and gradient parity at all three lengths. The
large-sequence memory saving is real, but its current backward kernel needs
more tiling and parallelism before it can replace the materialized fast path.

## UtauTTS speech-timing adoption (2026-10-04)

`SpeechTiming` composes the eight-block PyTorch Target on the float32 graph:
three phone embeddings of width 32 plus 4 or 15 continuous inputs, 1x1 input
convolution, dilated kernel-5 convolution/LayerNorm/exact GELU/dropout/residual
blocks, and an 80-channel 1x1 output convolution. Forward, NaN-masked L1,
all parameter gradients, clipping and an AdamW step pass a generated PyTorch
fixture on CPU and CUDA. A separate OneCycle test checks the PyTorch schedule
at six positions. The CUDA fixture asserts that forward output remains in a
device buffer.

An exported v1 cache contained 700 utterances and 286,260 frames. The exporter
recorded Python's seed-0 shuffle; both trainers used the same 30 validation
utterances and ran 6,000 CUDA steps at batch 16 and 400 frames. Gograd's
training-loop wall time was 66.477 s with best L1 0.285379 at step 4500.
PyTorch's measured `train()` loop took 73.625 s with best L1 0.284177 at step
3750. The quality difference is 0.42%, inside the 2% goal. Their initialization,
sampling and dropout random streams differ, so this is comparable architecture
and validation data rather than an identical optimization trace. A separate
full PyTorch script invocation took 81.862 s and reported best L1 0.2844.

The Go checkpoint loaded through the unchanged UtauTTS `speechtiming.LoadTCN`
and produced a one-frame, 80-bin prediction. PyTorch's safetensors reader also
accepted its metadata and 37 named tensors. The v2 context model can be built,
but a context feature cache was unavailable for a full training run. Optimizer
state, atomic checkpoint replacement and general-purpose data prefetch remain
open before broader adoption. `docs/API.md` lists the proposed v0.1 API; no
tag has been created.

## Variable-length CUDA pool fix (2026-10-04)

The UtauTTS frame-intonation trainer already releases its graph and owned
tensors each update. A reproducer with 64 different padded lengths kept active
buffers near 11 MB but grew the old exact-size, unlimited cache to
6,154,440,208 bytes (6,165,368,020 bytes reserved). With size classes and
bounded caching, the same 64 steps ended at 762,475,264 cached and
773,425,152 reserved bytes. In a 200-step run the cache stayed at that level;
after lowering the cap to 256 MiB, another 64 steps ended at 265,028,352
cached and 275,978,240 reserved bytes. Active buffers stayed at 10,949,888
bytes. A cudaMalloc allocation failure now releases cached blocks and retries
once. Host transfer slices are kept alive through the DLL call.

For fixed-shape Acoustic training, the final three paired CUDA runs measured
medians of 77.3 ms before and 77.8 ms after, a 0.7% difference. Earlier pairs
were slower for both binaries as device load changed. Remaining pool work includes
multi-device isolation and an optional driver-level free/total memory query.

## Irodori-TTS v4.1-Small inference inventory (2026-10-08)

The read-only reference is `irodori_tts` in UtauTTS's Irodori checkout. Its
v4.1-Small safetensors file has 714 F32 tensors and embeds both model and
ModernBERT configuration JSON. The model is 12 DiT blocks at width 1280 with
20 heads, a 32-dimensional latent, text and caption context width 512,
speaker context width 768, speaker patch size 4, and AdaLN rank 192. The
pretrained text and caption backbone is the 25-layer, width-768
`sbintuitions/modernbert-ja-310m` (HF Transformers 5.12.1 configuration).
The codec is a separate `weights.pth`, not safetensors. The following is the
complete runtime path and the gap against gograd as of this inventory:

| Stage | Reference behavior | gograd status / required work |
| --- | --- | --- |
| Text input | Ordered Japanese substitutions, bracket stripping, NFKC; HF fast tokenizer from `tokenizer.json`, BOS prepend, right padding/truncation to 256 text or 512 caption tokens | Normalization has a 12-case reference fixture; HF tokenizer JSON and token-ID fixtures remain open. The normalization function itself preserves ordinary whitespace. |
| Text/caption encoding | ModernBERT shared backbone, alternating full and 128-token sliding attention, two RoPE theta values (160000 and 10000), GELU MLP, RMS/normalization details, residual MLP projectors | General attention, linear, embedding, GELU and layer norm exist; ModernBERT architecture, sliding mask, RoPE variants, and checkpoint naming are missing. |
| Reference audio | WAV load, mono mix, resample to 48 kHz, AudioTools loudness normalization to -16 dB and peak limiting, deterministic DACVAE encode (encoder plus quantizer mean), patch/mask/clip/concat of up to eight reference clips | WAV/resampler/loudness parity and DACVAE encoder are missing. Latent reshape alone is straightforward. |
| Speaker context | ReferenceLatentEncoder: projected 32-dim latents, 4-token speaker patches, transformer self-attention with RoPE, gated attention, SwiGLU and RMSNorm | Gograd attention is reusable conceptually, but gated attention, RMSNorm, RoPE and model module are absent from its graph. CPU standalone RMSNorm/RoPE primitives now have PyTorch fixtures. |
| Duration | 14 text features, text token states, speaker/caption pooling/fusion, 3-layer token-sum dual AdaRN-zero duration predictor; `expm1` and frame-to-seconds clamp | Feature construction now has CPU fixture parity. Predictor, pooling, fusion and length parity are open. |
| DiT | Timestep cosine/sine embedding (512), latent input/output projections, 12 blocks with joint self+text+speaker+caption attention; Q/K RMSNorm, RoPE on half the heads, gated projection, context KV cache, SwiGLU and Low-Rank AdaLN with three shift/scale/gate low-rank paths | Timestep embedding, RMSNorm, full-head RoPE and Low-Rank AdaLN standalone CPU primitives have fixtures. Half-head RoPE, multi-segment masks, cached KV, SwiGLU and complete blocks require implementation and layer-by-layer fixtures. Existing generic MHA does not have these exact semantics. |
| RF sampler | Seeded PyTorch normal latent, 40-step Euler 0.999-to-0 grid (optional sway), independent/joint/alternating text/speaker/caption CFG in a time window, optional truncation, temporal score rescale and speaker KV scaling | Schedule and temporal score rescale have CPU fixture parity. PyTorch RNG equivalence, all CFG modes and the complete Euler loop are open. |
| Decoder/output | Unpatch latents, optional flat-tail trim, DACVAE decoder with deterministic message path and disabled codec watermark branch, trim to target sample count, SilentCipher `IRDTS` watermark, WAV write | DACVAE decoder and `.pth` loading, tail trim, watermark model, and audio parity are open. A generated WAV must not be called a drop-in reference output until SilentCipher is handled. |

`autograd.OpenSafeTensorFile` now validates a safetensors F32 index and reads
one named tensor at a time; `Module.LoadSafeTensorsStream` validates all state
names/shapes before copying. This avoids the existing all-file host read for
the multi-GB checkpoint. A small tensor from the real 714-tensor file was
dumped with `tools/gen_irodori_checkpoint_fixture.py` for exact byte/value
parity. The model modules are not yet built, so the full state dict cannot be
loaded into a runnable graph. CUDA work must stay below about 2 GB until the
other batch-generation job releases the 8 GB device.

Parity fixtures are generated with `tools/gen_irodori_fixture.py` from the
read-only Python implementation; this script needs no checkpoint. The current
CPU primitives and checkpoint indexing are foundations for a future Go-only
runtime, not a working TTS synthesizer. Whole-model numerical parity, a Go
inference CLI, real WAV output, and CPU/CUDA throughput comparison remain open.

Measured CPU fixture errors: RMSNorm and three duration-feature rows have
zero max-absolute error; full-head RoPE and sway schedule each have
`1.1920929e-7`; timestep embedding and temporal score rescale each have
`5.96046448e-8`; Low-Rank AdaLN output and gate have `2.38418579e-7` and
`5.96046448e-8`; linear schedule has zero error. The selected real checkpoint
tensor has zero max-absolute and relative error. A 256x1024 RMSNorm microbenchmark
on the i5-12400 measured 504,573 ns/op for Go (100 iterations) and a
285,800 ns/op median for PyTorch (100 iterations, one thread). These are
separate microbenchmarks, not a full-model speed comparison; no CUDA benchmark
was run while the device was occupied.
