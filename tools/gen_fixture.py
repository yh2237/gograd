#!/usr/bin/env python3
"""Generate a single-step TCN fixture for the Go reference implementation.

The model, centered loss and delta loss mirror the reference PyTorch
definitions so that the Go engine is checked against them. The fixture is
written in float64 so the comparison is limited by summation order, not dtype.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FrameIntonationTCN(nn.Module):
    def __init__(self, inputs: int, hidden: int, dilations=(1, 2, 4)):
        super().__init__()
        self.input = nn.Linear(inputs, hidden)
        self.layers = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, dilation=int(d)) for d in dilations]
        )
        self.output = nn.Linear(hidden, 1)
        self.dilations = tuple(int(d) for d in dilations)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        state = torch.tanh(self.input(values)).transpose(1, 2)
        for layer, dilation in zip(self.layers, self.dilations):
            convolved = layer(F.pad(state, (dilation, dilation)))
            state = torch.tanh(state + convolved)
        return self.output(state.transpose(1, 2)).squeeze(-1)


def centered(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    centers = []
    for row in range(values.shape[0]):
        selected = values[row][mask[row]]
        centers.append(selected.median() if selected.numel() else values[row].new_zeros(()))
    center = torch.stack(centers).unsqueeze(1)
    return values - center


def sequence_loss(predicted, targets, mask, bounded_target, delta_weight, target_scale):
    if not bool(mask.any()):
        return predicted.sum() * 0.0
    predicted = centered(predicted, mask)
    targets = centered(targets, mask)
    if bounded_target is not None:
        low, high = bounded_target
        low /= max(1.0, target_scale)
        high /= max(1.0, target_scale)
        targets = targets.clamp(float(low), float(high))
    absolute = F.smooth_l1_loss(predicted[mask], targets[mask])
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if not bool(pair_mask.any()):
        return absolute
    predicted_delta = predicted[:, 1:] - predicted[:, :-1]
    target_delta = targets[:, 1:] - targets[:, :-1]
    delta = F.smooth_l1_loss(predicted_delta[pair_mask], target_delta[pair_mask])
    return absolute + float(delta_weight) * delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="testdata/tcn_step.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inputs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--time", type=int, default=7)
    parser.add_argument("--low-cents", type=float, default=-250.0)
    parser.add_argument("--high-cents", type=float, default=250.0)
    parser.add_argument("--delta-weight", type=float, default=0.35)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = FrameIntonationTCN(args.inputs, args.hidden, args.dilations).double()

    def initial_params():
        return {
            "input_weight": model.input.weight.detach().cpu().tolist(),
            "input_bias": model.input.bias.detach().cpu().tolist(),
            "layers": [
                {
                    "dilation": dilation,
                    "weight": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                }
                for dilation, layer in zip(model.dilations, model.layers)
            ],
            "output_weight": model.output.weight.detach().cpu().tolist(),
            "output_bias": model.output.bias.detach().cpu().tolist(),
        }

    before = initial_params()
    values = torch.randn(args.batch, args.time, args.inputs, dtype=torch.float64)
    targets = torch.randn(args.batch, args.time, dtype=torch.float64) * 120.0

    # Variable-length rows exercise padding and the pair mask.
    lengths = [args.time, args.time - 2, args.time - 4]
    mask = torch.zeros(args.batch, args.time, dtype=torch.bool)
    for row, length in enumerate(lengths):
        mask[row, : max(1, length)] = True
    targets = targets * mask
    values = values * mask.unsqueeze(-1)

    predicted = model(values)
    loss = sequence_loss(
        predicted, targets, mask, (args.low_cents, args.high_cents), args.delta_weight, 1.0
    )
    loss.backward()

    def grads(tensor):
        return tensor.grad.detach().cpu().tolist()

    raw_grads = {
        "input_weight": grads(model.input.weight),
        "input_bias": grads(model.input.bias),
        "layers": [
            {"weight": grads(layer.weight), "bias": grads(layer.bias)}
            for layer in model.layers
        ],
        "output_weight": grads(model.output.weight),
        "output_bias": grads(model.output.bias),
    }

    learning_rate = 0.002
    weight_decay = 1e-5
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    clipped_grads = {
        "input_weight": grads(model.input.weight),
        "input_bias": grads(model.input.bias),
        "layers": [
            {"weight": grads(layer.weight), "bias": grads(layer.bias)}
            for layer in model.layers
        ],
        "output_weight": grads(model.output.weight),
        "output_bias": grads(model.output.bias),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.step()

    def params():
        return {
            "input_weight": model.input.weight.detach().cpu().tolist(),
            "input_bias": model.input.bias.detach().cpu().tolist(),
            "layers": [
                {
                    "dilation": dilation,
                    "weight": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                }
                for dilation, layer in zip(model.dilations, model.layers)
            ],
            "output_weight": model.output.weight.detach().cpu().tolist(),
            "output_bias": model.output.bias.detach().cpu().tolist(),
        }

    fixture = {
        "seed": args.seed,
        "inputs": args.inputs,
        "hidden": args.hidden,
        "dilations": args.dilations,
        "batch": args.batch,
        "time": args.time,
        "low_cents": args.low_cents,
        "high_cents": args.high_cents,
        "delta_weight": args.delta_weight,
        "values": values.tolist(),
        "targets": targets.tolist(),
        "mask": mask.tolist(),
        "params": before,
        "predicted": predicted.detach().cpu().tolist(),
        "loss": float(loss.detach()),
        "grads": raw_grads,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_norm": max_norm,
        "total_norm": float(total_norm),
        "clipped_grads": clipped_grads,
        "updated_params": params(),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"wrote {output} (loss={float(loss):.10f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
