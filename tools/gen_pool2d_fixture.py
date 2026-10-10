"""Generate PyTorch NCHW pooling forward/VJP fixtures; Go tests need only JSON."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def values(shape, phase):
    return torch.tensor([math.sin(i * .19 + phase) * .4 for i in range(math.prod(shape))],
                        dtype=torch.float32).reshape(shape).requires_grad_()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "testdata/pool2d.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = [
        ("max_default_stride", "max", [2, 3, 5, 7], dict(KernelSize=[2, 3])),
        ("max_overlap", "max", [2, 2, 5, 6], dict(KernelSize=[3, 2], Stride=[1, 1], Padding=[1, 1])),
        ("max_dilated_ceil", "max", [1, 2, 6, 7], dict(KernelSize=[3, 2], Stride=[2, 2], Padding=[1, 1], Dilation=[2, 2], CeilMode=True)),
        ("max_ceil_correction", "max", [1, 1, 4, 4], dict(KernelSize=[2, 2], Stride=[3, 3], Padding=[1, 1], CeilMode=True)),
        ("avg_default_stride", "avg", [2, 3, 5, 7], dict(KernelSize=[2, 3])),
        ("avg_include_pad", "avg", [2, 2, 4, 5], dict(KernelSize=[3, 3], Stride=[2, 2], Padding=[1, 1], CeilMode=True)),
        ("avg_exclude_pad", "avg", [2, 2, 4, 5], dict(KernelSize=[3, 3], Stride=[2, 2], Padding=[1, 1], CeilMode=True, ExcludePad=True)),
        ("avg_override", "avg", [1, 2, 5, 6], dict(KernelSize=[3, 2], Stride=[1, 2], Padding=[1, 1], CeilMode=True, ExcludePad=True, DivisorOverride=7)),
        ("avg_ceil_partial", "avg", [1, 2, 2, 3], dict(KernelSize=[3, 2], Stride=[3, 2], CeilMode=True)),
        ("avg_ceil_correction", "avg", [1, 1, 4, 4], dict(KernelSize=[2, 2], Stride=[3, 3], Padding=[1, 1], CeilMode=True)),
        ("adaptive_nondivisible", "adaptive", [2, 3, 5, 7], dict(OutputSize=[3, 4])),
        ("adaptive_global", "adaptive", [2, 2, 5, 6], dict(OutputSize=[1, 1])),
        ("adaptive_upsample", "adaptive", [1, 2, 2, 3], dict(OutputSize=[4, 5])),
    ]
    result = []
    for name, kind, shape, o in cases:
        x = values(shape, .3)
        if kind == "adaptive":
            y = F.adaptive_avg_pool2d(x, o["OutputSize"])
        else:
            args_common = (x, o["KernelSize"], o.get("Stride"), o.get("Padding", [0, 0]))
            if kind == "max":
                y = F.max_pool2d(*args_common, dilation=o.get("Dilation", [1, 1]), ceil_mode=o.get("CeilMode", False))
            else:
                y = F.avg_pool2d(*args_common, ceil_mode=o.get("CeilMode", False),
                                count_include_pad=not o.get("ExcludePad", False), divisor_override=o.get("DivisorOverride"))
        upstream = values(list(y.shape), 1.1).detach()
        (y * upstream).sum().backward()
        flat = lambda t: t.detach().reshape(-1).tolist()
        result.append(dict(name=name, kind=kind, x_shape=shape, y_shape=list(y.shape), options=o,
                           x=flat(x), y=flat(y), upstream=flat(upstream), dx=flat(x.grad)))
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=result), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(result)} cases")


if __name__ == "__main__":
    main()
