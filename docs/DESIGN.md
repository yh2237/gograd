# General tensor and autograd design

## Contract and current slice

`autograd` is a define-by-run float32 graph. A tensor has a logical shape,
row-major strides, a device, data, an optional accumulated gradient, and a
backward closure. The current implementation materializes contiguous results;
transpose and permute copy data. Its CUDA mode validates CUDA availability but
stages the new operators through host float32 arrays. This is a correctness
slice, not a competitive GPU implementation. The existing `tensor` and `nn`
packages retain their device-resident CUDA kernels.

The public design uses `Tensor{Shape, Strides, Device, Data, Grad}` for now.
The eventual storage object will own CPU or CUDA memory separately from the
view: shape, strides, offset and dtype. It will track sharing and release to
per-device pools after the last consumer. Dtypes will start with float32,
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

An op registry should select a kernel by `(op, dtype, device, layout)`.
Implementations share shape inference and gradient rules. CPU starts with
pure Go and blocked SGEMM; CUDA should dispatch cuBLAS and NVRTC kernels via
the existing DLL bindings. Host staging is only a temporary fallback and
should be explicit in profiling. Use CUDA streams and event lifetime tracking
before asynchronous frees. Pools should be bounded and keyed by device and
size class; graph capture requires stable allocations.

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
next `nn.Module` contract should expose ordered named parameters, buffers,
children, `state_dict`, `load_state_dict`, train/eval and device/dtype moves.
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

1. Add a device-resident storage abstraction, registry and CUDA kernels for
   elementwise, reductions, normalization, embedding and graph Conv1d; reuse
   `cuda`, `kernels`, `tensor.SGEMMOp` and cuBLAS. Make tape recording local to
   an execution context. Port `nn.Linear` and `nn.Conv1d` to graph operators,
   then make their explicit backward methods compatibility wrappers.
2. Move `gputcn` layers and its sequence loss onto the same operators. Retain
   its optimized CUDA graph path as a compiled execution plan. Add module
   state dictionaries and safetensors read/write, followed by dataset loaders.
3. Add LayerNorm, attention and transformer blocks, more indexing and
   reduction ops, mixed precision and loss scaling. Capture the general path
   with CUDA graphs and a reuse planner. Add CPU SIMD kernels selected at
   runtime with pure Go fallbacks. Validate against PyTorch on each dtype and
   shape family, and profile end-to-end steps before removing legacy paths.

The present prototype does not meet the GPU performance objective. In
particular, `Conv1dGEMM` and all normalization run on the host even when the
tensor device is CUDA; there is no device-resident graph storage yet.
