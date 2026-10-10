"""Generate CPU PyTorch BatchNorm output, VJP and state fixtures (Go-only tests)."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def values(shape, phase, offset=0., scale=.4):
    return torch.tensor([offset + math.sin(i * .19 + phase) * scale for i in range(math.prod(shape))],
                        dtype=torch.float32).reshape(shape)


def flat(t):
    return None if t is None else t.detach().reshape(-1).tolist()


def to_nc(x, last):
    return x.movedim(-1, 1) if last else x


def from_nc(x, last):
    return x.movedim(1, -1) if last else x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "testdata/batchnorm.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = [
        ("nc_training", [4, 3], True, False, True, True, True, .1),
        ("temporal", [3, 2, 5], True, False, True, True, True, .2),
        ("nchw", [2, 3, 4, 5], True, False, True, True, True, .1),
        ("rank5", [2, 3, 2, 3, 4], True, False, True, True, True, .5),
        ("rank6", [2, 2, 2, 2, 2, 2], True, False, True, True, True, .1),
        ("channels_last_temporal", [2, 5, 3], True, True, True, True, True, .1),
        ("channels_last_image", [2, 4, 5, 3], True, True, True, True, True, .1),
        ("no_affine_no_tracking", [2, 3, 4, 5], True, False, False, False, False, .1),
        ("weight_only", [2, 3, 4], True, False, True, False, True, .1),
        ("bias_only", [2, 3, 4], True, False, False, True, True, .1),
        ("frozen_running_values", [2, 3, 4], True, False, True, True, True, 0.),
        ("momentum_one", [2, 3, 4], True, False, True, True, True, 1.),
        ("eval_nchw", [2, 3, 4, 5], False, False, True, True, True, .1),
        ("eval_singleton", [1, 3], False, False, True, True, True, .1),
        ("eval_channels_last", [1, 4, 5, 3], False, True, False, False, True, .1),
        ("block_tail", [3, 2, 257], True, False, True, True, True, .1),
        ("channels_last_block_tail", [3, 257, 2], True, True, True, True, True, .1),
    ]
    result = []
    for name, shape, training, last, affine_w, affine_b, tracking, momentum in cases:
        channels = shape[-1] if last else shape[1]
        x = values(shape, .3).requires_grad_()
        w = values([channels], .7, .8, .2).requires_grad_() if affine_w else None
        b = values([channels], .9, 0., .1).requires_grad_() if affine_b else None
        mean = values([channels], .4, 0., .2) if tracking else None
        variance = values([channels], .6, 1.2, .1) if tracking else None
        initial_mean, initial_var = flat(mean), flat(variance)
        y = from_nc(F.batch_norm(to_nc(x, last), mean, variance, w, b, training, momentum, 1e-5), last)
        upstream = values(shape, 1.1)
        (y * upstream).sum().backward()
        result.append(dict(name=name, x_shape=shape, options=dict(Training=training, ChannelsLast=last, Eps=1e-5, Momentum=momentum),
                           x=flat(x), w=flat(w), b=flat(b), initial_mean=initial_mean, initial_var=initial_var,
                           y=flat(y), upstream=flat(upstream), dx=flat(x.grad), dw=flat(w.grad) if w is not None else None,
                           db=flat(b.grad) if b is not None else None, running_mean=flat(mean), running_var=flat(variance)))
    sequences = []
    for name, cumulative, disabled, last in [("tracked", False, False, False), ("cumulative", True, False, False),
                                             ("untracked", False, True, True)]:
        model = torch.nn.BatchNorm2d(2, momentum=None if cumulative else .25, track_running_stats=not disabled)
        w, b = values([2], .7, .8, .2), values([2], .9, 0., .1)
        with torch.no_grad():
            model.weight.copy_(w); model.bias.copy_(b)
        steps = []
        for step, (training, no_grad) in enumerate([(True, False), (True, True), (True, False), (False, False)]):
            shape = [1, 2, 1, 1] if not training and not disabled else [2 + step % 2, 2, 3, 4]
            if last:
                shape = [shape[0], shape[2], shape[3], shape[1]]
            x = values(shape, .3 + step * .7).requires_grad_()
            model.train(training); model.zero_grad()
            with torch.set_grad_enabled(not no_grad):
                y = from_nc(model(to_nc(x, last)), last)
            upstream = values(shape, 1.1)
            if not no_grad:
                (y * upstream).sum().backward()
            steps.append(dict(training=training, no_grad=no_grad, x_shape=shape, x=flat(x), y=flat(y), upstream=flat(upstream),
                              dx=flat(x.grad), dw=flat(model.weight.grad), db=flat(model.bias.grad),
                              running_mean=flat(model.running_mean), running_var=flat(model.running_var),
                              count=flat(model.num_batches_tracked)))
        sequences.append(dict(name=name, cumulative=cumulative, disabled=disabled, channels_last=last, w=flat(w), b=flat(b), steps=steps))
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=result, sequences=sequences), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(result)} cases, {len(sequences)} state sequences")


if __name__ == "__main__":
    main()
