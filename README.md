# gograd

A small reverse-mode autograd library and a frame-level intonation TCN, written
in Go. The engine computes in CPU float64.

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
a `Graph` and replay it with one launch. Freed buffers are reused from an
internal size pool, which cuts cudaMalloc/cudaFree traffic; `ReleasePool`
returns that memory to the driver. On other platforms every call reports
`ErrUnavailable`.

The `tensor` and `nn` packages are the general path: a shape-aware GPU tensor,
modules with explicit forward and backward, a shared AdamW optimizer and MSE
loss. A small MLP trains on synthetic data through them, and the TCN-specific
`gputcn` package stays as the specialized path.

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

## Commands

```
go test ./...
go vet ./...
go run ./cmd/tcn-train
go run ./cmd/gputcn-train
go test ./gputcn/ -run XXX -bench .
```

`tcn-train` fits a tiny synthetic corpus on the CPU and writes
`out/synthetic-tcn.json`. `gputcn-train` does the same with the GPU forward and
backward passes, prints per-phase timings, and writes `out/gputcn-synthetic.json`.
Passing `-graph` captures the whole step with `cuda.Capture` and replays it, and
`-profile N` reports GPU time per phase using CUDA events. Both are
demonstrations of the engine, not real training recipes.

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
```

The `cuda` tests run only when the CUDA runtime and cuBLAS DLLs are loadable.
On Windows, put the CUDA `bin\x64` directory on `PATH`, for example:

```
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH"
go test ./cuda/
```

## Layout

- `tensor.go`, `ops.go`, `conv.go`, `loss.go` — the autograd engine
- `model.go` — `FrameIntonationTCN`
- `optim.go` — AdamW and gradient clipping
- `export.go` — runtime JSON layout
- `cuda/` — CUDA runtime, cuBLAS and NVRTC binding, streams and graph capture
- `kernels/` — custom GPU kernels (activation, convolution, loss, AdamW) on top of `cuda`
- `tensor/` — shape-aware float32 GPU tensor
- `nn/` — modules, AdamW optimizer and MSE loss built on `tensor` and `kernels`
- `gputcn/` — GPU float32 TCN forward, backward and AdamW
- `cmd/tcn-train` — synthetic CPU training example
- `cmd/gputcn-train` — synthetic GPU training example with timings
- `cmd/gputcn-fit` — trains on a prepared dataset
- `tools/gen_fixture.py` — PyTorch fixture generator
- `testdata/` — committed fixtures

## Not present in this code

A general GPU tensor type integrated with the autograd engine; the GPU path is
specific to the TCN. Also absent: float32 training on the CPU, AMP, arbitrary
strides/broadcasting, additional models, and loading other frameworks'
checkpoints.

## License

MIT. See `LICENSE`.
