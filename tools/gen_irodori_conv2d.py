"""Conv2d / BatchNorm2d / gated-layer reference for gograd/irodori.

Mirrors the silentcipher Layer so the Go port can be checked layer by layer
without Python at test time. No checkpoint is needed: the weights are random
with a fixed seed.
"""

import json
from pathlib import Path

import numpy as np
import torch


def layer_output(kind, x, conv_w, conv_b, gate_w=None, gate_b=None, gamma=None,
                  beta=None, mean=None, var=None, eps=1e-5):
    main = torch.nn.functional.conv2d(x, conv_w, conv_b, padding=1 if conv_w.shape[2] == 3 else 0)
    if kind == "gated":
        main = main * torch.sigmoid(
            torch.nn.functional.conv2d(x, gate_w, gate_b, padding=1 if gate_w.shape[2] == 3 else 0))
    if gamma is not None:
        main = torch.nn.functional.batch_norm(main, None, None, gamma, beta, training=True, eps=eps)
    return main


def main():
    rng = np.random.default_rng(17)
    blob = bytearray()
    index = []
    cases = [
        # (in_channels, out_channels, kernel, bins, frames)
        (1, 32, 3, 9, 5),
        (32, 32, 3, 7, 4),
        (96, 96, 3, 5, 3),
        (96, 1, 1, 5, 3),
        (1, 1, 3, 4, 4),
    ]
    for in_channels, out_channels, kernel, bins, frames in cases:
        torch.manual_seed(in_channels * 1000 + out_channels + kernel)
        x = torch.from_numpy((rng.standard_normal((1, in_channels, bins, frames)) * 0.5).astype(np.float32))
        conv_w = torch.randn(out_channels, in_channels, kernel, kernel) * 0.3
        conv_b = torch.randn(out_channels) * 0.1
        gate_w = torch.randn(out_channels, in_channels, kernel, kernel) * 0.3
        gate_b = torch.randn(out_channels) * 0.1
        gamma = torch.rand(out_channels) + 0.5
        beta = torch.randn(out_channels) * 0.2

        plain = layer_output("plain", x, conv_w, conv_b)
        gated = layer_output("gated", x, conv_w, conv_b, gate_w, gate_b)
        normed = layer_output("gated", x, conv_w, conv_b, gate_w, gate_b, gamma, beta)

        start = len(blob)
        blob += x.numpy().astype("<f4").tobytes()
        for tensor in (conv_w, conv_b, gate_w, gate_b, gamma, beta):
            blob += tensor.numpy().astype("<f4").tobytes()
        for tensor in (plain, gated, normed):
            blob += tensor.numpy().astype("<f4").tobytes()
        # Sizes stay small so the fixture can record them inline.
        index.append({
            "in": in_channels, "out": out_channels, "kernel": kernel,
            "bins": bins, "frames": frames, "offset": start, "bytes": len(blob) - start,
        })
        print(f"case {in_channels}->{out_channels} k={kernel} {bins}x{frames}: "
              f"plain {plain.shape[2:]} gated+named {normed.shape[2:]}")
    Path("testdata/irodori_conv2d.bin").write_bytes(bytes(blob))
    Path("testdata/irodori_conv2d.json").write_text(
        json.dumps({"cases": index}, separators=(",", ":")) + "\n", encoding="utf-8")
    print("blob", len(blob))


if __name__ == "__main__":
    main()
