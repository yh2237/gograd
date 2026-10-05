# gograd

A small reverse-mode autograd library and a frame-level intonation TCN, written
in Go. The reference engine computes in CPU float64; the general `tensor` and
`nn` path trains in float32 on CPU or CUDA.

## Status

The engine is a CPU reference implementation in float64. It covers exactly the
operations the TCN and its loss need:

- `Linear`, `tanh`, elementwise add/sub/scale, `clamp`
- symmetric zero-padded dilated `Conv1d` (kernel 3, per-layer dilation)
- transpose and reshape, adjacent-frame difference
- mask gather, per-row median centering, Smooth L1 loss
- AdamW with decoupled weight decay and global gradient-norm clipping
- export of the `frame_pitch` JSON consumed by the Go runtime

The `cuda` package binds the CUDA runtime, cuBLAS and NVRTC DLLs on Windows
through the system loader, so no cgo or C compiler is needed. It provides device
queries, device buffers with explicit `Free`, streams, events, a row-major
`SgemmRowMajor` wrapper, an im2col + SGEMM `Conv1dForward`, and NVRTC-compiled
kernels loaded through the CUDA driver. It can capture a sequence of work into
a `Graph` and replay it with one launch. Freed buffers are reused from a
size-class pool capped at 2 GiB by default, so variable sequence lengths do
not retain every exact allocation size. `cuda.MemoryStats` reports live,
cached, reserved and peak bytes; `SetPoolCacheLimit` changes the cap, and
`ReleasePool` returns cached memory to the driver. On other platforms CUDA
operations report `ErrUnavailable` and `MemoryStats` returns zero counters.

The `autograd` package is a define-by-run float32 graph with rank-six views,
N-D broadcasting and reductions, batched matmul, embedding with padding,
Conv1d, normalization, softmax, log-softmax, cross-entropy, deterministic
dropout, scaled attention, masked losses, clipping, AdamW, and OneCycle LR.
The op registry lists the CPU and CUDA implementation for each public
differentiable tensor operation and selects its device backend; a source-level
test rejects an operation that bypasses dispatch. Pre-norm
`TransformerEncoderLayer` and batch-first `MultiheadAttention` compose these
ops. `Module` supports named state and F32 safetensors files compatible with
PyTorch exports.
Reusable `EmbeddingLayer`, `Conv1dLayer`, `LayerNormLayer`, `LinearLayer`,
`DropoutLayer`, and `GELULayer` modules compose the UtauTTS speech-timing
`Target` as `autograd.SpeechTiming`. Its checkpoint names and weight layouts
match PyTorch and UtauTTS's `speechtiming.LoadTCN`. The
`cmd/speech-timing-train` command trains from an exported safetensors feature
cache and saves the best validation weights with UtauTTS metadata. In v1.2.0,
it also saves a resumable training checkpoint containing current/best weights,
AdamW moments and hyperparameters, OneCycle state, split, and PCG sampler state.
`Module.Train` recursively propagates train/eval mode to registered children;
`WindowSampler` and `IndexLoader` expose restorable sampling state. See
[`docs/API.md`](docs/API.md) for the training surface.

CUDA batched matmul uses one strided cuBLAS call for regular and broadcast
batches. Attention combines batched GEMMs with fused masked softmax kernels.
An optional tiled CUDA path uses online softmax, recomputes scores in backward,
and never stores a full score matrix; it supports an additive broadcast mask.
The GEMM path remains the automatic choice when its scratch estimate is below
1 GiB, because it is faster at the tested sequence lengths. CUDA layer norm
and the encoder feed-forward bias/GELU/residual use fused kernels. Warp softmax
handles arbitrary axes. Optional `-bf16` caches BF16 GEMM operands on device,
carries BF16 shadows through views, and fuses BF16 output writes into pointwise
kernels. GEMMEx uses FP32 accumulation; gradients and master weights remain
FP32. The default remains FP32.
`autograd.Acoustic` composes the reference acoustic model and is checked
against a PyTorch training-step fixture on CPU and CUDA. CUDA tensors and
gradients now live in device buffers; graph operators use cuBLAS and NVRTC
kernels, and clipping and AdamW stay on device. CUDA callers pin an OS thread
with `autograd.NewCUDAContext` for the driver context. CPU Conv1d uses blocked
SGEMM with an AVX2/FMA 8x8 microkernel and a pure Go fallback; batched matmul uses
the same implementation. The Acoustic step can be captured into a CUDA graph
with fixed device indices and a device AdamW step counter. See
[`docs/DESIGN.md`](docs/DESIGN.md) for the architecture and migration plan.

The `tensor` and `nn` packages are the earlier general float32 path. Select
`tensor.CPU` or `tensor.CUDA` at construction with `NewOn`, `FromHostOn`,
`NewLinearOn`, and `NewConv1dOn`. The original constructors still select CUDA.
CPU matrix multiplication uses multithreaded blocked SGEMM with AVX2/FMA
when available and a pure Go fallback. Modules
support `[batch,time,channels]`: Linear, same-padded dilated Conv1d, ReLU,
LeakyReLU, tanh-approximation GELU, Tanh, Sequential, and Residual. The two
masked losses accept a `[batch,time]` frame weight (zero for padding) and
optional channel weights. Each divides by the sum of element weights, and
returns a loss and output gradient. AdamW clips the global gradient norm and
uses decoupled weight decay on either device.

General CUDA Conv1d uses im2col/col2im kernels and cuBLAS for both passes.
Activations, residual additions, masked losses, and AdamW run on the device.
The trainer keeps fixed loss weights on the device and downloads only the
requested scalar loss during each step.

The `gputcn` package runs the TCN forward and backward passes on the GPU in
float32. The forward output matches the PyTorch float64 reference to about
5e-08 and the parameter gradients to about 1e-05. The loss and AdamW, including
the global norm clip, run on the GPU. `SequenceLossGrad` reproduces the CPU
`SequenceLoss` (per-row median centering, clamped target, Smooth L1 and the
delta term) with one block per row, and is checked against `LossGrad`. A GPU
training loop reduces the synthetic loss.

## Verification

The reference engine is checked against PyTorch, which remains the comparison
baseline:

- One forward, loss, backward, gradient clipping and one AdamW step match a
  PyTorch fixture (`testdata/tcn_step.json`) to about 1e-16 for the forward and
  gradients in float64.
- The median follows `torch.median`: the lower middle value for an even count,
  with its gradient shared equally among elements equal to the median.
- Gradients are also checked against central finite differences, independently
  of PyTorch.
- The general float32 modules and masked losses have CPU finite-difference
  checks. `testdata/nn_conv_step.json`, generated by `tools/gen_nn_fixture.py`,
  checks forward, masked loss, backward and one AdamW update against PyTorch
  on CPU and on CUDA when available.
- `testdata/acoustic_step.json`, generated by `tools/gen_acoustic_fixture.py`,
  checks the Acoustic output, loss, every parameter gradient, clipping norm,
  and one AdamW step on both devices.
- `tools/gen_transformer_fixture.py` generates PyTorch values in a temporary
  directory during tests. Softmax, log-softmax, cross-entropy, dropout,
  layer norm, broadcast batched matmul, attention, padding embedding, and a
  two-layer transformer stack are checked for outputs and gradients on both
  devices; the stack also checks one AdamW update. CUDA tests assert kernel
  launches. A PyTorch-produced safetensors file loads in Go and round-trips
  back to PyTorch. A CUDA graph test compares two Acoustic replays with eager
  steps, including updated optimizer weights.
- The registry inventory test checks every public differentiable operation
  for a registry entry and a dispatch call. Removing one dispatch was verified
  to make the test fail. Tiled attention checks masked output and gradients,
  including sequence lengths 256, 1024 and 4096, against the materialized
  CUDA path. CUDA residency tests assert device kernel launches; BF16 graph
  tests check captured training steps.
- `tools/gen_speech_timing_fixture.py` compares the full 8-block Target with
  PyTorch on CPU and CUDA: forward, NaN-masked L1, every parameter gradient,
  clipping norm, and one AdamW step. CUDA output residency is asserted.

## Commands

```
go test ./...
go vet ./...
go run ./cmd/tcn-train
go run ./cmd/gputcn-train
go run ./cmd/nn-conv-train -device cpu -hidden 128 -kernel 5 -steps 3
go run ./cmd/nn-conv-train -device cpu -hidden 128 -kernel 5 -steps 5 -cpuprofile "$env:TEMP/gograd-nn-cpu.pprof"
go test ./tensor -run '^$' -bench 'Benchmark(SGEMM|NaiveSGEMM)$' -benchtime=1x -cpu=12
go test ./gputcn/ -run XXX -bench .
go run ./cmd/acoustic-bench -device cpu
go run ./cmd/acoustic-bench -device cuda -warmup 1 -gpu-profile
go run ./cmd/acoustic-bench -device cuda -warmup 2 -graph
go run ./cmd/acoustic-bench -device cuda -warmup 2 -memory-stats
python tools/bench_acoustic.py --device cuda --threads 12
python tools/bench_acoustic.py --device cuda --threads 12 --graph
go run ./cmd/transformer-bench -device cuda -warmup 1
go run ./cmd/transformer-bench -device cuda -warmup 2 -graph -bf16
python tools/bench_transformer.py --device cuda --threads 12 --warmup 1
python tools/bench_transformer.py --device cuda --warmup 2 --graph --bf16
go run ./cmd/attention-bench -seq 4096 -algorithm materialized
go run ./cmd/attention-bench -seq 4096 -algorithm flash
python tools/export_speech_timing_features.py --input D:/project/UtauTTS/out/speech-timing-target/features.pt --output $env:TEMP/gograd-speech-timing/features.safetensors
go run ./cmd/speech-timing-train -cache $env:TEMP/gograd-speech-timing/features.safetensors -steps 6000 -device cuda -out $env:TEMP/gograd-speech-timing/model.safetensors
```

`tcn-train` fits a tiny synthetic corpus on the CPU and writes
`out/synthetic-tcn.json`. `gputcn-train` does the same with the GPU forward and
backward passes, prints per-phase timings, and writes `out/gputcn-synthetic.json`.
Passing `-graph` captures the whole step with `cuda.Capture` and replays it, and
`-profile N` reports GPU time per phase using CUDA events. Both are
demonstrations of the engine, not real training recipes.

`nn-conv-train` trains a six-block residual Conv1d network on padded synthetic
sequences (input 74, output 72, dilations 1,2,4,1,2,4), prints full training
step times, and accepts `-device cuda` when CUDA is available. It defaults to
masked L1 loss; `-loss mse` selects masked MSE. Set `GOMAXPROCS=12` for the
12-thread benchmark and use `-cpuprofile` to write a Go CPU profile.

Medians of three paired, warmed full training steps on the RTX 3060 Ti and
12 CPU threads (`GOMAXPROCS=12` for Go, `torch.set_num_threads(12)`):

| Model and runtime | CPU FP32 | CUDA FP32 eager | CUDA FP32 graph | CUDA BF16 eager | CUDA BF16 graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acoustic, PyTorch | 2,506 ms | 58.5 ms | 53.7 ms | 47.1 ms | 42.6 ms |
| Acoustic, gograd | 3,876 ms | 82.3 ms | 82.0 ms | 63.8 ms | 64.1 ms |
| Transformer, PyTorch | 1,292 ms | 25.9 ms | 23.9 ms | 13.5 ms | 12.0 ms |
| Transformer, gograd | 2,276 ms | 27.2 ms | 27.8 ms | 22.4 ms | 23.2 ms |

Acoustic is hidden 384, batch 24, 400 frames, 10 blocks. Transformer is four
pre-norm layers with model width 256, four heads, feed-forward width 1024,
batch 16, sequence 256. Steps include forward, backward, global-norm clip,
and AdamW. Acoustic graph replay includes a scalar loss readback; its device
step counter updates AdamW bias correction each launch. CPU timings varied
substantially during measurement, so these medians are more useful than a
single step but should not be treated as stable throughput. BF16 values use
PyTorch autocast and gograd's FP32-authoritative BF16 shadow path; their scope
still differs. The
implementations use different seeded random values and are comparable by
architecture and dimensions, not identical loss values.

Transformer CPU is 1.76x PyTorch, meeting the 2.5x target; Acoustic CPU is
1.55x, narrowly missing its 1.5x target. Transformer CUDA FP32 eager is
1.05x PyTorch. BF16 eager is 1.66x PyTorch for the transformer and 1.35x for
Acoustic. Graph replay is similar to eager for gograd; device work dominates
the launch cost. Profiling adds synchronization and should not be compared
to the table.

CUDA event medians for a separate attention forward/backward benchmark
(batch 1, four heads, head dimension 64; three runs):

| Sequence | Materialized step | Tiled step | Materialized peak extra | Tiled peak extra |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.366 ms | 0.766 ms | 3.50 MiB | 1.01 MiB |
| 1024 | 1.601 ms | 9.606 ms | 50.00 MiB | 4.03 MiB |
| 4096 | 18.916 ms | 127.889 ms | 776.00 MiB | 16.12 MiB |

Peak extra is the maximum live CUDA buffer increase above the input baseline,
including loss and gradient buffers. Tiled attention trades speed for much
lower scratch use at long sequences. Output and gradient parity against the
materialized path passes at all three lengths; a separate test covers an
additive broadcast mask and its gradient.

### UtauTTS speech-timing target

The v1 feature cache contains 700 utterances and 286,260 frames. The exporter
stores the Python seed-0 shuffle in its safetensors metadata, so the Go and
PyTorch trainers use the same 30-utterance validation split. Both ran 6,000
steps with batch 16, 400-frame windows, AdamW, OneCycle, clip 1.0 and CUDA.

| Trainer | Best validation L1 | Best step | Training-loop wall time |
| --- | ---: | ---: | ---: |
| UtauTTS PyTorch | 0.284177 | 3750 | 73.625 s |
| gograd | 0.285379 | 4500 | 66.477 s |

These measurements used the v0.1 trainer/sampler. The Go checkpoint is 0.42% above the measured PyTorch L1 and loads with
UtauTTS's `speechtiming.LoadTCN`; its `Predict` returned one 80-bin frame in
the read-only integration check. PyTorch's complete script, including cache
loading, export and startup, took 81.862 s in a separate run and reached
0.2844. Different initialization, sampling and dropout streams mean the
training trajectories are not identical. The 15-feature context path is
implemented but had no context cache available for a full run.

### Resumable speech-timing training (v1.2.0)

`--steps` is the complete OneCycle schedule, not the number of additional
updates after resuming. Use a fresh `--out` path on each invocation; the best
inference weights are retained inside the training checkpoint and re-exported.

```sh
go run ./cmd/speech-timing-train --cache features.safetensors \
  --steps 6000 --valid 30 --seed 0 --device cpu \
  --out out/best-first.safetensors --checkpoint out/training.safetensors \
  --checkpoint-every 250 --stop-after 2000

go run ./cmd/speech-timing-train --cache features.safetensors \
  --steps 6000 --valid 30 --seed 0 --device cpu \
  --out out/best-resumed.safetensors --checkpoint out/training.safetensors \
  --resume out/training.safetensors
```

Ctrl+C requests a stop after the current update and writes the training
checkpoint. A forced termination can only resume from the last periodic save.
Keep cache contents, seed, validation count, context mode, total steps,
batch/window size and evaluation interval unchanged. Inference weights alone
are not resumable checkpoints. CPU resume tests compare final files byte for
byte; CUDA tests check all tensors within `1e-6` absolute error and exact
sampler/scheduler state because atomic gradient reductions can reorder sums.

The window sampler now uses a restorable PCG stream and includes the final
complete window. A fresh v1.2.0 run does not reproduce the old Go RNG trajectory.
This command's resumable state is new; legacy `nn`/`gputcn` trainers are unchanged.

### Variable-length CUDA memory

In a 64-step, batch-8 speech-timing training reproduction with changing frame
lengths, active allocation bytes stayed nearly constant while the old
exact-size pool retained 6.15 GB. The size-class, bounded pool reduced the
same workload's cache to 0.76 GB (decimal bytes below). A 200-step test checks
the default cap and also repeats training with a 256 MiB cap.

| Allocator after 64 steps | Live bytes | Cached bytes | Reserved bytes |
| --- | ---: | ---: | ---: |
| Previous | 10,927,812 | 6,154,440,208 | 6,165,368,020 |
| Current | 10,949,888 | 762,475,264 | 773,425,152 |

The fixed-shape Acoustic CUDA step had paired median times of 77.3 ms before
and 77.8 ms after in the final three-pair check. Earlier pairs were slower
for both binaries as device load changed. Captured graph
allocations stay reserved until `Graph.Close` and are excluded from the cache
limit while the graph is live.

`gputcn-fit` trains on a prepared dataset (frame features, targets and mask)
instead of the synthetic corpus:

```
go run ./cmd/gputcn-fit -write-synthetic out/prepared.json
go run ./cmd/gputcn-fit -dataset out/prepared.json -epochs 30
```

The dataset is JSON with `feature_names` and records that store sparse features
as `rows`, `cols` and `vals`, plus `targets` (cents) and a `mask`. Records are
split into train and validation by a hash of the id.

To regenerate the PyTorch fixture (requires `torch`):

```
python tools/gen_fixture.py
python tools/gen_nn_fixture.py
```

The `cuda` tests run only when the CUDA runtime and cuBLAS DLLs are loadable.
On Windows, put the CUDA `bin\x64` directory on `PATH`, for example:

```
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH"
go test ./cuda/
```

## Layout

- `autograd/layers.go`, `autograd/speech_timing.go` — reusable modules and the
  UtauTTS speech-timing Target composition
- `cmd/speech-timing-train`, `tools/export_speech_timing_features.py`,
  `tools/gen_speech_timing_fixture.py` — feature bridge, trainer, and parity
- `docs/API.md` — proposed v0.1 API and experimental boundaries
- `autograd/cpu_binary.go`, `autograd/cuda_flash_attention.go`,
  `autograd/cuda_autocast.go` — optimized CPU gradients, tiled attention,
  and cached BF16 device buffers
- `cmd/attention-bench` — CUDA attention time and live buffer comparison
- `autograd/views.go`, `autograd/registry.go`, `autograd/transformer*.go`,
  `autograd/safetensors.go` — views, elementwise dispatch, transformer ops and modules, checkpoints
- `cmd/transformer-bench`, `tools/bench_transformer.py` — four-layer encoder timings
- `tools/gen_transformer_fixture.py`, `tools/safetensors_fixture.py` — PyTorch parity references

- `tensor.go`, `ops.go`, `conv.go`, `loss.go` — the autograd engine
- `model.go` — `FrameIntonationTCN`
- `optim.go` — AdamW and gradient clipping
- `export.go` — runtime JSON layout
- `cuda/` — CUDA runtime, cuBLAS and NVRTC binding, streams and graph capture
- `kernels/` — custom GPU kernels (activation, convolution, loss, AdamW) on top of `cuda`
- `tensor/` — shape-aware float32 CPU/CUDA tensor and blocked CPU SGEMM
- `autograd/` — define-by-run graph, operators, and Acoustic model
- `nn/` — CPU/CUDA modules, masked losses, AdamW, and JSON checkpoints
- `gputcn/` — GPU float32 TCN forward, backward and AdamW
- `cmd/tcn-train` — synthetic CPU training example
- `cmd/gputcn-train` — synthetic GPU training example with timings
- `cmd/gputcn-fit` — trains on a prepared dataset
- `cmd/nn-conv-train` — residual Conv1d float32 synthetic training example
- `tools/gen_fixture.py`, `tools/gen_nn_fixture.py` — PyTorch fixture generators
- `tools/gen_acoustic_fixture.py`, `tools/bench_acoustic.py` — Acoustic parity and timing
- `docs/DESIGN.md` — architecture and migration roadmap
- `testdata/` — committed fixtures

## General model JSON

`nn.Save(writer, model)` writes a versioned JSON object; `nn.Load(reader,
model)` checks the layer configuration, parameter names, and shapes before
loading. `layers` is the ordered forward list. Each entry has `name`, `type`
(`linear`, `conv1d`, `activation`, or `residual`), `in_channels`,
`out_channels`, and where applicable `kernel`, `dilation`, `activation`, and
`slope`. A residual's `layers` list describes its inner sequence. `parameters`
is an ordered array of `{name,shape,values}` with row-major float32 values.
Linear weights are `[out,in]`, biases `[out]`; Conv1d weights are
`[out,in,kernel]` in cross-correlation order and biases are `[out]`. Frames
are `[batch,time,channels]` and zero same-padding extends by
`dilation*(kernel-1)/2` on each side. Names are stable paths such as
`model.1.inner.0.weight`. Optimizer state is not included.

## Not present in this code

The graph still materializes many contiguous FP32 results. BF16 is optional
and keeps FP32 authoritative storage; FP16, loss scaling, a dtype/layout keyed
kernel registry, and a general prefetching data loader are not present. The
speech-timing trainer reads an exported feature cache and samples windows on
the host. The registry selects CPU or CUDA implementations but does not yet
support runtime kernel plugins. The
recording flag and attention algorithm selection are process-wide. Tiled
attention currently supports head dimensions up to 128 and is slower than the
materialized GEMM path at the measured sequence lengths. GPU inference callers
should release unused graphs with `ReleaseGraph`. The older `nn` path still
uses explicit module backward methods.

## License

MIT. See `LICENSE`.
