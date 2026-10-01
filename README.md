# gograd

A small reverse-mode autograd library and a frame-level intonation TCN, written
in Go. All computation is CPU float64.

## Status

This repository contains the CPU reference implementation in float64. It covers
exactly the operations the TCN and its loss need:

- `Linear`, `tanh`, elementwise add/sub/scale, `clamp`
- symmetric zero-padded dilated `Conv1d` (kernel 3, per-layer dilation)
- transpose and reshape, adjacent-frame difference
- mask gather, per-row median centering, Smooth L1 loss
- AdamW with decoupled weight decay and global gradient-norm clipping
- export of the `frame_pitch` JSON consumed by the Go runtime

There is no GPU backend.

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
```

`tcn-train` fits a tiny synthetic corpus and writes `out/synthetic-tcn.json`. It
is a demonstration of the engine, not a real training recipe.

To regenerate the PyTorch fixture (requires `torch`):

```
python tools/gen_fixture.py
```

## Layout

- `tensor.go`, `ops.go`, `conv.go`, `loss.go` — the autograd engine
- `model.go` — `FrameIntonationTCN`
- `optim.go` — AdamW and gradient clipping
- `export.go` — runtime JSON layout
- `cmd/tcn-train` — synthetic training example
- `tools/gen_fixture.py` — PyTorch fixture generator
- `testdata/` — committed fixtures

## Not present in this code

GPU execution, float32 training, AMP, arbitrary strides/broadcasting, additional
models, and loading other frameworks' checkpoints.

## License

MIT. See `LICENSE`.
