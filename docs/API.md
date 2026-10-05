# v0.1 API proposal

The first intended consumer is UtauTTS's speech-timing target trainer. A
`v0.1.0` tag is proposed after integration review; no tag is created here.

## Supported training surface

| Package | API | Contract |
| --- | --- | --- |
| `tensor` | `Device`, `CPU`, `CUDA` | Select the backend when constructing tensors and models. |
| `autograd` | `NewCUDAContext`, `CUDAContext.Close` | Hold a CUDA context on the calling OS thread for the training step. |
| `autograd` | `New`, `Tensor.ToHost`, `Tensor.Backward`, `Tensor.ReleaseGraph`, `Tensor.Close` | Float32 tensors, explicit readback and lifetime control. |
| `autograd` | `NewSpeechTiming`, `SpeechTiming.Forward`, `Train`, `Parameters`, `Module` | The UtauTTS Target architecture and PyTorch-compatible parameter names. `Forward` accepts flattened phone IDs and a `[batch,time,C]` tensor. |
| `autograd` | `NewEmbeddingLayer`, `NewConv1dLayer`, `NewLayerNormLayer`, `NewLinearLayer`, `DropoutLayer`, `GELULayer`, `Sequential` | Reusable graph modules. Convolution activations are `[batch,time,channels]`; weights are `[out,in,kernel]`. |
| `autograd` | `Module.StateDict`, `LoadStateDict`, `SaveSafeTensors`, `SaveSafeTensorsMetadata`, `LoadSafeTensors` | Named float32 state. Metadata values are strings. The speech-timing model emits UtauTTS loader names and shapes. |
| `autograd` | `MaskedLoss(pred,target,false)`, `ClipGradNorm`, `NewAdamW`, `NewOneCycle` | NaN-frame masked L1, clipping, AdamW, and PyTorch two-phase cosine OneCycle LR. |
| `autograd` | `AdamW.State`, `LoadState`, `SaveSafeTensors`, `LoadSafeTensors` | Resumable AdamW moments and step count on CPU and CUDA. |
| `autograd` | `SetTraining`, `IndexLoader` | Training-mode propagation for modules such as `DropoutLayer`; deterministic shuffled mini-batch indices. |
| `cuda` | `MemoryStats`, `ResetAllocationPeak`, `SetPoolCacheLimit`, `ReleasePool` | Query active, cached, and reserved bytes; bound or release reusable CUDA allocations. |

`NewSpeechTiming(4, device, seed)` builds v1; `NewSpeechTiming(15, device,
seed)` builds the context model. The final seed argument initializes weights
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

## Experimental and compatibility limits

- General autograd operators, CUDA graph capture, BF16 autocast, device buffer
  pooling, and the op registry remain experimental. Device and dtype handling
  is currently float32-first; BF16 uses optional shadow buffers.
- `NoGrad`, BF16 mode, and attention selection use process-wide state. Do not
  concurrently mutate these settings or run independent training contexts.
- Optimizer moments and the AdamW step count are serialized with
  `SaveSafeTensors`/`LoadSafeTensors`; the OneCycle scheduler position is not,
  so restore it from the saved step count.
- Safetensors loading validates names, shapes, and byte ranges. Atomic saves,
  overlapping-range rejection, and a file-size limit are still open.
- The Go and PyTorch training runs use different random streams for sampling,
  initialization, and dropout. Fixed initial weights and explicit dropout
  masks are used for operation-level parity; full-run validation is compared
  by measured quality rather than identical trajectories.
