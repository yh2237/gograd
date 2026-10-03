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
runtime-selected AVX2/FMA 4x8 microkernel and pure Go fallback; CPU elementwise
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
captures. Pools should eventually be
bounded and keyed by device as well as size.

Warp reductions fuse softmax forward and backward. The attention fast path
uses batched QK and PV GEMMs around fused masked scale/softmax forward and
backward kernels. Layer norm has direct fused row kernels, and the encoder's
feed-forward path fuses bias with GELU and residual addition on CUDA when
dropout is zero. BF16 autocast currently applies to GEMM operands only:
conversion runs in a device kernel, GEMMEx accumulates into FP32, and all
weights and optimizer moments remain FP32. Conversion buffers come from the
CUDA pool. This mode is optional, process-wide, and requires more parity
coverage before being used as a default.

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
3. Add FlashAttention-style tiled attention that avoids materialized scores,
   broaden mixed precision to normalization/elementwise kernels, and add fp16
   loss scaling. Extend the general CUDA
   graph capture API beyond fixed Acoustic and transformer batches. Extend CPU SIMD beyond SGEMM while retaining
   pure Go fallbacks. Validate against PyTorch on each dtype and shape family
   before removing legacy paths.

Current single-run measurements and methods are in README. A four-layer encoder
(model 256, four heads, feed-forward 1024, batch 16, sequence 256) measures
28.7 ms CUDA eager versus PyTorch 23.5 ms, and 28.3 ms captured versus 24.0 ms.
BF16 measures 27.3/25.8 ms eager/graph versus PyTorch 13.1/12.3 ms; conversion
and remaining FP32 pointwise work limit its gain. CPU measures 2.39 s versus
PyTorch 0.477 s. CPU profiling attributes 34.8% of samples to AVX SGEMM,
7.1% to tensor storage indexing, 7.1% to binary backward, and 6.3% to exp.
Acoustic measures 4.19 s CPU versus PyTorch 1.93 s, 84.3 ms CUDA eager versus
60.9 ms, and 80.8 ms graph replay versus 53.8 ms. Graph replay spends about
1.0 ms in launch submission and 79.8 ms waiting for device completion;
the GEMM and convolution workload dominates, so capture saves little.
