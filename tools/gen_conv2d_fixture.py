"""Generate checkpoint-free Conv2d forward/VJP fixtures for CPU and CUDA tests.

Run from any directory: python tools/gen_conv2d_fixture.py [--output FILE].
Only fixture generation needs PyTorch; Go tests read the committed JSON.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def values(shape, phase):
    n = math.prod(shape)
    # Compute in Python float64 then narrow once: independent of torch RNGs.
    return torch.tensor([math.sin(i * .19 + phase) * .4 for i in range(n)],
                        dtype=torch.float32).reshape(shape).requires_grad_()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "testdata/conv2d.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = [
        ("batched", [2, 2, 5, 6], [3, 2, 3, 3], [1, 1], [1, 1], [1, 1], 1, True),
        ("grouped", [2, 4, 6, 7], [6, 2, 2, 3], [2, 1], [1, 2], [2, 1], 2, True),
        ("depthwise_multiplier", [2, 3, 7, 6], [6, 1, 3, 2], [1, 2], [2, 1], [2, 2], 3, True),
        ("even_no_bias", [1, 2, 6, 8], [2, 2, 2, 4], [2, 2], [0, 1], [1, 1], 1, False),
        ("pointwise", [2, 4, 3, 5], [6, 2, 1, 1], [1, 1], [0, 0], [1, 1], 2, False),
        ("tile_tail", [1, 2, 23, 25], [3, 2, 3, 3], [1, 1], [1, 1], [1, 1], 1, True),
        ("wide_padding", [1, 1, 2, 3], [2, 1, 3, 2], [2, 3], [4, 3], [1, 2], 1, True),
    ]
    result = []
    for name, xs, ws, stride, padding, dilation, groups, biased in cases:
        x, w = values(xs, .3), values(ws, .9)
        bias = values([ws[0]], .7) if biased else None
        y = F.conv2d(x, w, bias, stride, padding, dilation, groups)
        upstream = values(list(y.shape), 1.1).detach()
        (y * upstream).sum().backward()
        flat = lambda t: t.detach().reshape(-1).tolist() if t is not None else None
        result.append(dict(name=name, x_shape=xs, w_shape=ws, y_shape=list(y.shape),
                           options=dict(Stride=stride, Padding=padding, Dilation=dilation, Groups=groups),
                           x=flat(x), w=flat(w), b=flat(bias), upstream=flat(upstream),
                           y=flat(y), dx=flat(x.grad), dw=flat(w.grad), db=flat(bias.grad) if biased else None))
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=result), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(result)} cases")


if __name__ == "__main__":
    main()
