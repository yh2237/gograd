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
unused CUDA inference graphs. Views currently keep the parent tensor alive,
and a chained view may materialize its parent. A separate reference-counted
storage object remains future work. Dtypes start with float32,
then add float16 and bfloat16 compute with float32 accumulation and int64
indices. Operators must reject mixed devices and unsupported dtype pairs.
Operators accept non-contiguous inputs by materializing them before kernel
dispatch. Views keep their base storage alive.

## Graph and operators

Each op records its parents and a vector-Jacobian product when recording is
enabled. Backward topologically traverses the graph in reverse, accumulating
all contributions to leaves. Intermediate gradients are discarded unless
`RetainGrad` is called. `Detach` returns data without history; `NoGrad`
disables recording for its callback. These controls should become a per-tape
context when concurrent training is added: the present package-wide recording
flag is not goroutine safe. In-place mutation after forward must eventually
use version counters to reject stale saved tensors.

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
FP32. This mode is optional and process-wide. Full BF16 tensor storage and
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

## Migration and phases

1. Extend the operation registry from backend selection to callable kernels
   keyed by dtype and layout, and make views composable without intermediate
   materialization. Make tape recording local to an execution context. Port
   `nn.Linear` and `nn.Conv1d` to graph operators, then make their explicit
   backward methods compatibility wrappers.
2. Move `gputcn` layers and its sequence loss onto the same operators. Retain
   its optimized CUDA graph path as a compiled execution plan. Add general
   dataset loaders, optimizer state checkpoints, and recursive module
   train/eval propagation. `SpeechTiming` already toggles its dropout state.
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
