# v1.2.0 training API

The first consumer is UtauTTS's speech-timing target trainer. Inference
safetensors keep their existing names/layouts; resumable training checkpoints
use a separate `gograd-training-1` envelope.

## Supported training surface

| Package | API | Contract |
| --- | --- | --- |
| `tensor` | `Device`, `CPU`, `CUDA` | Select the backend when constructing tensors and models. |
| `autograd` | `NewCUDAContext`, `CUDAContext.Close` | Hold a CUDA context on the calling OS thread for the training step. |
| `autograd` | `New`, `Tensor.ToHost`, `Tensor.Backward`, `Tensor.ReleaseGraph`, `Tensor.Close` | Float32 tensors, explicit readback and lifetime control. |
| `autograd` | `RetainData`, `RetainGrad`, `BackwardWithOptions`, `BackwardOptions` | Shared-storage lifetimes, retained forward values/gradients and explicit derivative-history retention for multiple backward passes. |
| `autograd` | `NewExecutionContext`, `ExecutionContext.New`, `Zeros`, `Bind`, `BindModule`, `NoGrad`, `Recording`, `Tensor.ExecutionContext` | Explicit recording scope propagated by tensors and views. Independent CPU models can train/infer concurrently using separate contexts. |
| `autograd` | `ExecutionOptions`, `ExecutionContext.SetOptions`, `Options`, `Autocast` | Context-local BF16/attention policy, scoped precision and forward-policy snapshots for backward. |
| `autograd` | `NewSpeechTiming`, `SpeechTiming.Forward`, `Train`, `Parameters`, `Module` | The UtauTTS Target architecture and PyTorch-compatible parameter names. `Forward` accepts flattened phone IDs and a `[batch,time,C]` tensor. |
| `autograd` | `NewEmbeddingLayer`, `NewConv1dLayer`, `NewLayerNormLayer`, `NewLinearLayer`, `DropoutLayer`, `GELULayer`, `Sequential` | Reusable graph modules. Convolution activations are `[batch,time,channels]`; weights are `[out,in,kernel]`. |
| `autograd` | `Conv2d`, `Conv2dOptions`, `NewConv2dLayer`, `Conv2dLayer` | NCHW activations, `[out,in/groups,kH,kW]` weights, optional bias, grouped/depthwise convolution and CPU/CUDA gradients. |
| `autograd` | `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, pooling options and layers | NCHW pooling on CPU/CUDA with overlapping-window gradients, ceil mode and explicit averaging divisors. |
| `autograd` | `Module.StateDict`, `LoadStateDict`, `SaveSafeTensors`, `SaveSafeTensorsMetadata`, `LoadSafeTensors` | Named float32 state. Metadata values are strings. The speech-timing model emits UtauTTS loader names and shapes. |
| `autograd` | `MaskedLoss(pred,target,false)`, `ClipGradNorm`, `NewAdamW`, `NewOneCycle` | NaN-frame masked L1, clipping, AdamW, and PyTorch two-phase cosine OneCycle LR. |
| `autograd` | `AdamW.State`, `LoadState`, `SaveSafeTensors`, `LoadSafeTensors`, `Close` | Named parameter shapes, moments, hyperparameters and step count on CPU/CUDA; release optimizer-owned buffers. |
| `autograd` | `SaveTrainingCheckpoint`, `LoadTrainingCheckpoint`, `TrainingCheckpointMetadata` | Atomic model/buffer + AdamW + OneCycle checkpoint with caller metadata. All names/shapes/state are validated before copying into live objects. |
| `autograd` | `Module.Train`, `SetTraining` | Recursive train/eval propagation; `OnTrainingChange` connects stateful layers such as dropout to the module tree. |
| `autograd` | `IndexLoader.State`, `LoadState`, `WindowSampler.Batch`, `State`, `LoadState` | Restorable epoch order or PCG random windows, including short-sequence padding/context-drop decisions. |
| `cuda` | `MemoryStats`, `ResetAllocationPeak`, `SetPoolCacheLimit`, `ReleasePool` | Query active, cached, and reserved bytes; bound or release reusable CUDA allocations. |

`NewSpeechTiming(4, device, seed)` builds v1; `NewSpeechTiming(15, device,
seed)` builds the context model. `NewSpeechTimingWithPhones(phones, continuous,
device, seed)` builds the same architecture for a language-specific phone
vocabulary (for example UtauTTS English or Mandarin timing targets); the
embedding row count equals `phones`. The final seed argument initializes weights
with Go's random stream. Loading a PyTorch safetensors state provides exact
weight parity. `SpeechTiming.Forward(ids, cont, dropoutSeed)` uses a deterministic
CPU/CUDA hash mask while training and disables dropout after `Train(false)`.

The training command consumes the `gograd-speech-timing-features-1`
safetensors cache written by `tools/export_speech_timing_features.py`. Its
`I64` phone indices and `F32` continuous inputs and targets are named
`00000.ids`, `00000.cont`, and `00000.target`, and metadata stores record IDs.
This cache format is experimental; it is a bridge from UtauTTS's trusted
PyTorch `.pt` cache, not a general dataset API.

`cuda.MemoryStats()` reports physical bytes owned by gograd's CUDA allocator:
`LiveBytes`, `CachedBytes`, `ReservedBytes`, `PeakLiveBytes`,
`PeakReservedBytes`, `CachedBuffers`, and `CacheLimitBytes`. The default cache
limit is 2 GiB. `SetPoolCacheLimit` evicts least-recently-used cached buffers
immediately when lowering it; live tensors and allocations reserved by a
captured graph remain allocated. `ResetAllocationPeak` resets peak counters
to the current levels. The counters do not include allocations made directly
by CUDA libraries outside gograd's pool. `Buffer.Size()` remains the requested
logical byte count even when its physical size class is larger.

## Execution contexts and gradient recording

Use `ExecutionContext` for independent executions. The context is carried by
tensors, propagated by operators and views, and preserved by `Detach`; no
goroutine-local state is used. A zero-value context records gradients.

```go
execution := autograd.NewExecutionContext()
model, err := autograd.NewLinearLayer(2, 3, tensor.CPU,
    rand.New(rand.NewSource(7)))
// Handle err.
_ = err
err = execution.BindModule(model.StateModule())
// Handle err. Bind every parameter/buffer before Forward or starting workers.
x, err := execution.New([]float32{1, 2}, []int{1, 2}, tensor.CPU, true)
// Handle err.
_ = err
loss := autograd.Sum(model.Forward(x), 0, 1)
err = loss.Backward()
// Handle err; read gradients before releasing the graph.
loss.ReleaseGraph()

execution.NoGrad(func() {
    prediction := model.Forward(x)
    // prediction.RequiresGrad is false; ToHost can read its values.
    prediction.ReleaseGraph()
})
// execution.Recording() is true again, including after a callback panic.
```

- `BindModule` covers registered parameters and buffers recursively and
  validates all leaves before changing any context. Bind before building a
  graph, including parameter-only branches (weight transpose, embedding, etc.).
  `Bind` supports an existing leaf; assigning a foreign context or binding an
  already-built graph is rejected. Parameter/optimizer values are not copied.
- An operation combining two different explicit contexts panics. Unbound
  constants may join a bound graph. For full isolation, bind all model state
  and construct all inputs/masks on the context, including non-gradient inputs.
- `execution.NoGrad` is local to that context and nests safely. It preserves
  factory `requiresGrad` arguments and does not alter existing graphs;
  Backward on a previously recorded loss is still allowed inside NoGrad.
  Backward on an unrecorded/constant scalar returns an error on CPU and CUDA.
- Package `autograd.New`/`Zeros` and `autograd.NoGrad` remain the compatibility
  API for unbound tensors. Package NoGrad controls a **shared default scope**;
  it does not disable explicit contexts. New code using an explicit context
  must call that context's NoGrad rather than the package function.
- NoGrad controls the entire context, not one goroutine. Use one context and
  independent model state per independently executing worker. Tensor values,
  gradients, optimizer state and train/eval switches are not safe to mutate
  concurrently on a shared model. Binding itself is an initialization step.
- Inference results keep lifetime parents so `ReleaseGraph` can reclaim CPU
  intermediates and CUDA auxiliaries; they do not retain derivative callbacks.
  Retain handles whose values must outlive graph reclamation. `Detach` makes independent storage,
  drops history and keeps context identity (including non-contiguous views).

This context isolates gradient recording, BF16 mode and attention selection.
It does not own CUDA streams/capture or allocation pools. CUDA
callers still hold `NewCUDAContext` on the calling OS thread, and independent
concurrent CUDA execution is not covered by this API. `cmd/cnn-train` uses the
explicit context API.

### Precision and attention policy

Each explicit context defaults to `ExecutionOptions{BF16Autocast: false,
AttentionAlgorithm: "auto"}`, including a zero-value context. It does not
inherit package-level `BF16Autocast` or `AttentionAlgorithm`. `Options` returns
a copy; use `SetOptions` to update it between forwards. Invalid attention names
return an error without changing the context; an empty name normalizes to `auto`.

```go
err = execution.SetOptions(autograd.ExecutionOptions{
    AttentionAlgorithm: "materialized",
})
// Handle err. Bind model state and construct inputs on execution as above.
var mixedLoss *autograd.Tensor
execution.Autocast(true, func() {
    mixedLoss = autograd.Sum(model.Forward(x), 0, 1)
})
// Precision has returned to FP32. On CUDA, backward still uses the BF16
// policy selected by eligible operations during the preceding forward.
err = mixedLoss.Backward()
// Handle err.
mixedLoss.ReleaseGraph()
```

- `Autocast(enabled, fn)` changes only this context's BF16 flag. Nested scopes
  and panics restore the previous precision; attention selection is unaffected.
  Like NoGrad, it affects the whole context rather than one goroutine. Independent
  workers need separate contexts and mutable model state.
- Each precision-sensitive operation snapshots its settings during forward.
  Its backward, including retained-history passes, keeps that policy even if
  the scope exits, `SetOptions` changes, or compatibility globals change.
  Captured CUDA graphs likewise retain their captured kernel/precision choices
  on replay. Changing settings requires a new capture to alter those choices.
- BF16 uses CUDA GEMM operands/shadows with FP32 accumulation, outputs,
  gradients, master weights and optimizer moments. CPU computations and
  unsupported CUDA operations such as Conv2d remain FP32. The tiled flash
  attention implementation currently computes in FP32 too.
- Attention names are `materialized`, `flash`, and `auto`. For compatible CUDA
  attention without active dropout, `materialized` uses GEMM/softmax and `flash`
  uses the tiled path (Q/K and V head dimensions at most 128). `auto` selects
  tiled attention when estimated materialized score scratch reaches 1 GiB.
  CPU, active dropout and non-fusible layouts use the composed operator path.
- Unbound tensors still use the package-level compatibility settings. These
  globals are shared; do not mutate them concurrently. Explicit contexts never
  consult them, and unbound constants may join a context-bound operation.
  Set context options between forwards: changing them partway through a
  multi-operation forward does not make that entire forward an atomic snapshot.

## Tensor storage and graph lifetimes

Tensor/view allocations are reference-counted independently of graph edges.
Closing a base handle does not return its allocation to the CPU/CUDA pool while
a dependent view or saved computation still uses it. Releasing one graph branch
does not reclaim an ancestor still used by another branch. This is explicit
lifetime management: abandoned graphs still need `ReleaseGraph`/`Close`.

- Factory leaves from `New`/`Zeros` (inputs, parameters, buffers) remain owned
  by the caller. `ReleaseGraph` reclaims intermediates and internal ephemeral
  leaves; persistent inputs/parameters require `Close`.
- `Close` terminates that handle's public use and is idempotent. Its allocation
  and saved state remain readable **internally** until dependent graph edges
  are done. A surviving view can still be read or used in backward, but the
  closed base's `ToHost`, `GradToHost` and `CopyFrom` return errors.
- `RetainData` pins forward values; `RetainGrad` pins values and intermediate
  gradients. Neither retains derivative history. Use `Close` to release those
  retained handles after reading them. A retained view can keep only its shared
  allocation after its base graph has been reclaimed.
- `Backward` consumes derivative callbacks and saved statistics. On CUDA it
  also reclaims unused intermediates automatically; on CPU forward values
  remain until `ReleaseGraph`. Read unretained CUDA outputs before backward.
- `BackwardWithOptions(BackwardOptions{RetainGraph: true})` keeps history and
  saved allocations for another pass on either backend. Leaf gradients
  accumulate; intermediate adjoints are fresh for propagation, while requested
  retained gradients accumulate without being propagated twice.
- A second backward through consumed history returns an error. Combining
  losses into a single scalar and running one backward needs no history
  retention. `ReleaseGraph` is still required when retained passes are done.

```go
x := autograd.Must([]float32{2, 3}, []int{2}, true)
shared := autograd.Mul(x, x)
left := autograd.Sum(shared, 0)
right := autograd.Sum(autograd.MulScalar(shared, 3), 0)
err := left.BackwardWithOptions(autograd.BackwardOptions{RetainGraph: true})
// Handle err; x's gradient is [4,6].
_ = err
err = right.Backward()
// Handle err; accumulated x gradient is [16,24].
left.ReleaseGraph()
right.ReleaseGraph()
x.Close()
```

Masked-loss targets and CUDA embedding index buffers are saved lifetime
dependencies even though they are not differentiated. They remain available
to retained backward after the caller closes its original handle. CPU
normalization statistics and im2col buffers likewise return to pools only when
history is consumed or released. BF16 shadows have shared leases and are
invalidated across aliases when their FP32 storage changes.

`CopyFrom` and eager AdamW updates increment a storage version shared by its
views. Backward validates recorded operand/output versions **before** changing
gradients and rejects modification since forward, including between retained
passes. The checks are conservative; some derivatives may not need every
operand. Direct CPU `Data` writes and opaque CUDA buffer writes/replays are
not version-tracked: do not change saved values through those paths during a
graph's lifetime. Initialize weights before building the graph.

`Data` slices and `Buffer()` are borrowed access, not independent owners. Do not
replace `Data`, Free a tensor-owned buffer, or use borrowed memory after its
handle ends. Keep Tensor/IndexBuffer values as pointers rather than copying
their ownership state. Independent CPU contexts may share read-only constants;
concurrent mutation/close/backward on shared tensors is not supported.

## General Conv2d and CNN training

`Conv2d(x, weight, bias, options)` performs zero-padded cross-correlation on
`[batch,channels,height,width]` input. Weights use PyTorch's
`[out_channels,in_channels/groups,kernel_height,kernel_width]` layout; pass
`nil` for bias or a tensor of shape `[out_channels]`. Kernel dimensions come
from the weights, and can be rectangular or even.

`Conv2dOptions` contains `[2]int` pairs in height/width order:

- `Stride` and `Dilation`: an all-zero pair defaults to `[1,1]`; otherwise
  both entries must be positive.
- `Padding`: non-negative symmetric zero-padding on each axis. The default
  is valid convolution with no padding.
- `Groups`: zero defaults to 1. Both channel counts must divide by groups.
  `Groups == in_channels` implements depthwise convolution, including an
  output-channel multiplier.

For each axis the output size is
`floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`.
Invalid device combinations, shapes or options panic at the operation boundary;
the layer constructor returns errors. Input, kernel and output dimensions must
be positive, with each tensor's element count bounded by the CUDA signed
32-bit indexing limit. Non-contiguous views are accepted and receive gradients
through their materialization.

```go
device := tensor.CPU // or tensor.CUDA while holding NewCUDAContext
rng := rand.New(rand.NewSource(7))
conv, err := autograd.NewConv2dLayer(4, 8, [2]int{3, 5},
    autograd.Conv2dOptions{
        Stride: [2]int{1, 2}, Padding: [2]int{1, 2}, Groups: 2,
    }, true, device, rng) // true registers a bias; false leaves Bias nil
// Handle err. conv.StateModule() registers "weight" and optionally "bias".
_ = err
root := autograd.Module{Children: []autograd.NamedModule{
    {Name: "conv", Module: conv.StateModule()},
}}
x, err := autograd.New(make([]float32, 2*4*8*10), []int{2, 4, 8, 10}, device, true)
// Handle err.
_ = err
y := conv.Forward(x) // [2,8,8,5]
loss := autograd.Mean(y, 0, 1, 2, 3)
err = loss.Backward() // input, weight and bias gradients
// Handle err; read outputs/gradients before releasing the graph.
loss.ReleaseGraph()
err = root.SaveSafeTensors("conv.safetensors")
// Handle err. Close input, parameters and optimizer-owned buffers when done.
```

CPU and CUDA both lower one batch/group at a time into spatial tiles. The
im2col/output tile pair is limited to 4 MiB (or one column if the kernel alone
exceeds that budget), with at most 512 output positions per tile. Backward
recomputes the lowering rather than saving a full im2col matrix. CPU uses the
shared SGEMM; CUDA uses cuBLAS plus device lowering/scatter kernels with no
host staging. Conv2d currently computes in FP32 even when BF16 autocast is
enabled. CUDA input-gradient scatter uses atomic additions and is numerically,
not bit-for-bit, reproducible.

`cmd/cnn-train` exercises two Conv2d/ReLU blocks, max downsampling, adaptive
global average pooling, a linear classifier, cross-entropy and AdamW on synthetic noisy vertical and
horizontal stripe images. Saving reconstructs a fresh model, reloads every
parameter exactly, and compares inference logits (CUDA reductions may reorder
sums). No external dataset or Python is needed:

```sh
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/cnn-train -device cuda -steps 60 -out out/cnn-cuda.safetensors
```

The `out` parent must exist. These are architecture/training demonstrations,
not image-recognition quality benchmarks. Committed Conv2d fixtures check all
output and VJP elements against PyTorch without requiring Python during Go
tests; regenerate with `python tools/gen_conv2d_fixture.py`.

## NCHW pooling

`MaxPool2d`, `AvgPool2d` and `AdaptiveAvgPool2d` accept float32
`[batch,channels,height,width]` inputs on CPU or CUDA. Non-contiguous views are
materialized and their gradients are mapped back to the original storage.
Input/output dimensions must be positive and total element counts must fit
signed 32-bit indexing, as for Conv2d. Invalid options/shapes panic before
materialization or allocating output storage.

```go
down := autograd.MaxPool2d(x, autograd.MaxPool2dOptions{
    KernelSize: [2]int{2, 2}, // default stride is also [2,2]
})
smoothed := autograd.AvgPool2d(down, autograd.AvgPool2dOptions{
    KernelSize: [2]int{3, 3}, Stride: [2]int{1, 1},
    Padding: [2]int{1, 1}, ExcludePad: true,
})
pooled := autograd.AdaptiveAvgPool2d(smoothed, [2]int{1, 1})
features := autograd.Reshape(pooled, pooled.Shape[0], pooled.Shape[1])
// features is [batch,channels]; compose with a LinearLayer and loss.
```

- `MaxPool2dOptions` has `KernelSize`, `Stride`, `Padding`, `Dilation` pairs
  and `CeilMode`. `AvgPool2dOptions` has `KernelSize`, `Stride`, `Padding`,
  `CeilMode`, `ExcludePad`, and `DivisorOverride`. Pairs are height/width.
  KernelSize is required. An all-zero stride defaults to KernelSize; an
  all-zero max-pool dilation defaults to `[1,1]`. Other stride/dilation entries
  must be positive. Padding is symmetric and between zero and half the kernel
  size on each axis. Average pooling has unit dilation.
- Output size on each axis is
  `floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`.
  CeilMode replaces floor with ceil, then removes a final window starting
  wholly in the right/bottom padding. Windows may extend past the input.
- Max padding acts as negative infinity. Equal maxima select the first valid
  row-major sample; NaNs select the last NaN. Selected indices are saved only
  when gradients are recorded and survive retained backward passes. A dilated
  window containing no valid sample returns negative infinity and contributes
  no input gradient. This API returns values, not an indices tensor.
- Average pooling counts zero padding in its divisor by default, matching
  PyTorch's `count_include_pad=true`. Set `ExcludePad: true` to count only
  valid input positions. For ceil-mode partial windows, positions beyond the
  requested padding are excluded even in the default mode. A positive
  `DivisorOverride` replaces the count; zero uses the normal count. Negative
  overrides are rejected.
- Adaptive pooling requires a positive `[output_height,output_width]`.
  Each bin starts at `floor(position*input/output)` and ends at
  `ceil((position+1)*input/output)`. Bins can overlap and output sizes may be
  larger than input sizes; backward adds every bin's contribution.
- `MaxPool2dLayer{Options: ...}`, `AvgPool2dLayer{Options: ...}` and
  `AdaptiveAvgPool2dLayer{OutputSize: ...}` implement `TensorLayer` for
  `Sequential`. They have no parameters, buffers or train/eval-dependent state.
  Bound input context identity and NoGrad behavior propagate normally.
- Pooling computes in FP32, including under BF16 autocast. CUDA uses device
  kernels without host staging, supports capture/replay, and accumulates
  overlapping gradients with atomic additions. Compare gradients numerically,
  rather than requiring bit-for-bit CUDA reproducibility.

Committed PyTorch fixtures cover every output and input VJP for rectangular
kernels, dilated max pooling, ceil-mode correction, include/exclude padding,
divisor override, non-divisible adaptive bins and adaptive upsampling.
Regeneration requires PyTorch (`python tools/gen_pool2d_fixture.py`); Go tests
use the JSON directly. Finite differences independently verify all three ops.

## Resuming a training run

```go
sampler, err := autograd.NewWindowSampler(frameCounts, 400, 16, 0.15, 0)
// Handle err; gather the returned windows into tensors on the chosen device.
windows := sampler.Batch()
_ = windows
sampleJSON, err := json.Marshal(sampler.State())
// Handle err; save only after an optimizer update and scheduler.Step(opt).
err = autograd.SaveTrainingCheckpoint("training.safetensors", &model.Module,
    opt, schedule, map[string]string{"sampler": string(sampleJSON)})
// Handle err; model/optimizer/scheduler must already be constructed.
metadata, err := autograd.LoadTrainingCheckpoint("training.safetensors",
    &model.Module, opt, schedule)
// Handle err; restore the stream before sampling another batch.
var state autograd.WindowSamplerState
err = json.Unmarshal([]byte(metadata["sampler"]), &state)
// Handle err.
err = sampler.LoadState(state)
// Handle err; opt.Close() releases its own buffers at the end of training.
```

Parameter order, names and shapes must match. Loading restores learning rate,
decay, betas and epsilon rather than silently using new defaults. Model mode,
dropout seeds, dataset identity and sampler state are the caller's responsibility;
`speech-timing-train` records/validates these and uses completed steps for the
deterministic dropout seed. `OneCycle` is serialized by the combined checkpoint;
separate inference or optimizer files are not a full training checkpoint.
For CUDA graphs, synchronize outstanding launches before saving and restore
state before capturing a new graph. The optimizer snapshot reads the captured
device counter rather than an outdated host step count.

## Experimental and compatibility limits

- General autograd operators, CUDA graph capture, BF16 autocast, device buffer
  pooling, and the op registry remain experimental. Device and dtype handling
  is currently float32-first; BF16 uses optional shadow buffers.
- The compatibility `NoGrad` scope is shared by unbound tensors. Use separate
  explicit contexts and independent model state for concurrent CPU execution.
  Unbound tensors use process-wide BF16/attention compatibility settings; do
  not mutate those globals concurrently. Explicit contexts own these settings.
  CUDA stream/capture isolation is not yet implemented.
- Combined checkpoints restore optimizer and OneCycle state. Old `nn`/`gputcn`
  trainers still use their existing checkpoint paths. Sampling runs on the host;
  a prefetching dataset loader is not implemented.
- Safetensors writes use a same-directory temporary file, sync and rename.
  Loading rejects invalid/overflowing shapes, overlapping ranges and headers
  above 16 MiB. The whole file is read into memory; there is no total file-size
  limit. Device transfer failures can still interrupt an otherwise validated load.
- The Go and PyTorch training runs use different random streams for sampling,
  initialization, and dropout. Fixed initial weights and explicit dropout
  masks are used for operation-level parity; full-run validation is compared
  by measured quality rather than identical trajectories.
