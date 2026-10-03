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

An elementwise op registry now pairs CPU forward/gradient rules with CUDA
opcodes for add/sub/mul/div and drives their dispatch. Other operators still
dispatch directly; a uniform registry keyed by `(op, dtype, device, layout)`
remains to be built. CPU uses blocked SGEMM with a
runtime-selected AVX2/FMA microkernel and pure Go fallback; CPU elementwise
ops and reductions use multiple cores, and size pools reuse buffers. CUDA
Conv1d uses im2col/col2im plus cuBLAS SGEMM. NVRTC kernels implement
elementwise broadcasting, shape transforms, embedding, reductions,
normalization, masked losses, softmax, cross-entropy, dropout, and clipping.
CUDA N-D indexing supports ranks through six for broadcasting, reduction,
and views. The CUDA driver context is tied to
an OS thread, so callers hold `NewCUDAContext` through the training step.
The graph currently uses one stream. Captured graphs reserve their temporary
pool allocations until `Graph.Close`, and cuBLAS is bound to the capture
stream. This does not yet support overlapping independent streams or concurrent
captures. Pools should eventually be
bounded and keyed by device as well as size.

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

1. Complete the op registry beyond elementwise operations and make views
   composable without intermediate materialization. The CUDA kernels for
   elementwise ops, reductions, normalization, embedding, attention primitives
   and graph Conv1d are implemented. Make tape recording local to an execution
   context. Port `nn.Linear` and `nn.Conv1d` to graph operators, then make
   their explicit backward methods compatibility wrappers.
2. Move `gputcn` layers and its sequence loss onto the same operators. Retain
   its optimized CUDA graph path as a compiled execution plan. Add dataset
   loaders, optimizer state checkpoints, and model train/eval propagation.
3. Fuse attention/softmax and normalization kernels, add more indexing and
   reduction ops, mixed precision and loss scaling. Extend the general CUDA
   graph capture API beyond fixed Acoustic batches. Extend CPU SIMD beyond SGEMM while retaining
   pure Go fallbacks. Validate against PyTorch on each dtype and shape family
   before removing legacy paths.

Current single-run measurements (12 CPU threads): Acoustic was 79.8 ms on
CUDA versus PyTorch's 58.2 ms, and 6.56 s on CPU versus PyTorch's 2.58 s.
One captured Acoustic replay took 79.8 ms including scalar loss readback
versus PyTorch's 55.1 ms;
the device AdamW step counter advances on each replay. A four-layer encoder
(model 256, four heads, feed-forward 1024, batch 16, sequence 256) took
89.9 ms on CUDA versus PyTorch's 24.6 ms and 4.00 s on CPU versus PyTorch's
1.04 s. The encoder still uses separate softmax and batched GEMM launches,
which is the main target for a fused attention path. An instrumented CUDA
step recorded 896 calls each to matmul forward/input-gradient/weight-gradient
SGEMM (about 25 ms in each category) and 13 ms in softmax backward. Event
profiling raised total step time to 175 ms. Measurements are single warmed
steps, not steady-state distributions.
