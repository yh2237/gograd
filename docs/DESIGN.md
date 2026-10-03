# General tensor and autograd design

## Contract and current slice

`autograd` is a define-by-run float32 graph. A tensor has a logical shape,
row-major strides, a device, data or a CUDA buffer, an optional accumulated
gradient, and a backward closure. The current implementation materializes
contiguous results; transpose and permute copy data. CUDA graph values,
gradients, optimizer moments and clipped gradients remain in device buffers.
Training uploads inputs and reads the scalar loss. Parity tests and explicit
state export also read tensors back for inspection.

The public design uses `Tensor{Shape, Strides, DType, Device}` with CPU slices
or CUDA buffers. `Backward` returns CUDA intermediates to the allocation
pool; `ReleaseGraph` releases CPU intermediates for reuse and also frees
unused CUDA inference graphs. Reshape shares storage. The eventual storage
object will separate views from ownership and support offsets and arbitrary
strides. Dtypes start with float32,
then add float16 and bfloat16 compute with float32 accumulation and int64
indices. Operators must reject mixed devices and unsupported dtype pairs.
Non-contiguous views will be permitted once kernels accept strides; reshape
will copy if required. Views keep their base storage alive.

## Graph and operators

Each op records its parents and a vector-Jacobian product when recording is
enabled. Backward topologically traverses the graph in reverse, accumulating
all contributions to leaves. Intermediate gradients are discarded unless
`RetainGrad` is called. `Detach` returns data without history; `NoGrad`
disables recording for its callback. These controls should become a per-tape
context when concurrent training is added: the present package-wide recording
flag is not goroutine safe. In-place mutation after forward must eventually
use version counters to reject stale saved tensors.

An op registry should eventually select a kernel by `(op, dtype, device,
layout)`. Dispatch is currently direct. CPU uses blocked SGEMM with a
runtime-selected AVX2/FMA microkernel and pure Go fallback; CPU elementwise
ops and reductions use multiple cores, and size pools reuse buffers. CUDA
Conv1d uses im2col/col2im plus cuBLAS SGEMM. NVRTC kernels implement
elementwise broadcasting, shape transforms, embedding, reductions,
normalization, masked losses and clipping. The CUDA driver context is tied to
an OS thread, so callers hold `NewCUDAContext` through the training step.
The graph currently uses one stream. Stream-aware lifetime tracking is needed
before cross-stream frees or graph capture. Pools should eventually be
bounded and keyed by device as well as size.

Broadcasting aligns dimensions from the right. Each pair must match or one
dimension must be 1. Backward sums over expanded dimensions. Matmul applies
this to batch dimensions, with the last two dimensions as matrices. Reduction
axes are explicit and removed from output shape; scalar output has shape
`[]`. Concat and slice use zero-based axes and half-open slice bounds.
Embedding backward scatter-adds repeated indices. Conv1d uses
`[batch,time,channels]`, weights `[out,in,kernel]`, odd kernel, cross
correlation, dilation and symmetric zero same-padding. GroupNorm computes
statistics over time and channels per sample and group. LayerNorm computes
statistics over the final dimension. Masked loss uses non-NaN target frames
and divides by all valid frame/channel elements, matching the Acoustic
reference. A zero-valid-element loss currently returns zero.

## Modules and training

`Acoustic` is the first graph-composed model. Its parameter names match the
PyTorch reference, and the small fixture tests all of their gradients. The
`Module` now registers ordered named parameters, buffers and children and
provides `StateDict` and `LoadStateDict`. Train/eval and device/dtype moves
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

For serialization, use safetensors-compatible tensor payloads and metadata:
an 8-byte little-endian header length, JSON descriptors with dtype, shape and
byte offsets, followed by contiguous payload. Put model architecture and
optimizer/scheduler metadata in a separate versioned manifest. Validate
shapes, ranges, duplicate names and size limits before allocation; write
atomically. Existing `nn` JSON checkpoints remain readable during migration.

## Migration and phases

1. Complete the device-resident storage abstraction and add an op registry.
   The CUDA kernels for elementwise ops, reductions, normalization, embedding
   and graph Conv1d are implemented. Make tape recording local to an execution
   context. Port `nn.Linear` and `nn.Conv1d` to graph operators, then make
   their explicit backward methods compatibility wrappers.
2. Move `gputcn` layers and its sequence loss onto the same operators. Retain
   its optimized CUDA graph path as a compiled execution plan. Add module
   state dictionaries and safetensors read/write, followed by dataset loaders.
3. Add attention and transformer blocks, more indexing and reduction ops,
   mixed precision and loss scaling. Capture the general path with CUDA
   graphs and a reuse planner. Extend CPU SIMD beyond SGEMM while retaining
   pure Go fallbacks. Validate against PyTorch on each dtype and shape family
   before removing legacy paths.

At the real Acoustic shape, the warmed full step measured 78.2 ms on the RTX
3060 Ti versus 53.0 ms for PyTorch CUDA; CPU measured 5.42 s versus 8.52 s
for PyTorch. The op registry, tape-local recording and migration of `nn` and
`gputcn` remain open. GPU event profiling shows convolution SGEMMs dominate
the step. CUDA graph capture, mixed precision and a reusable execution plan
are the next performance opportunities.
