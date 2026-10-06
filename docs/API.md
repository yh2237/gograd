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
| `autograd` | `NewSpeechTiming`, `SpeechTiming.Forward`, `Train`, `Parameters`, `Module` | The UtauTTS Target architecture and PyTorch-compatible parameter names. `Forward` accepts flattened phone IDs and a `[batch,time,C]` tensor. |
| `autograd` | `NewEmbeddingLayer`, `NewConv1dLayer`, `NewLayerNormLayer`, `NewLinearLayer`, `DropoutLayer`, `GELULayer`, `Sequential` | Reusable graph modules. Convolution activations are `[batch,time,channels]`; weights are `[out,in,kernel]`. |
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
- `NoGrad`, BF16 mode, and attention selection use process-wide state. Do not
  concurrently mutate these settings or run independent training contexts.
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
