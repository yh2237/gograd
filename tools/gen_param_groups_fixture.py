"""Generate PyTorch parameter-group updates for Go-only tests."""
import argparse
import json
from pathlib import Path

import torch


def build(kind, params, groups, gradients):
    tensors = [torch.nn.Parameter(torch.tensor(p["value"], dtype=torch.float32)) for p in params]
    lookup = {}
    for group in groups:
        for name in group["params"]:
            lookup[name] = group
    optimizer = torch.optim.SGD if kind == "sgd" else torch.optim.AdamW
    group_args = []
    for group in groups:
        members = [t for t, p in zip(tensors, params) if p["name"] in group["params"]]
        args = {"params": members}
        args.update({k: v for k, v in group.items() if k != "params"})
        group_args.append(args)
    options = dict(lr=group_args[0]["lr"])
    if kind == "sgd":
        options.update(momentum=.9, nesterov=True)
    optim = optimizer(group_args, **options)
    steps = []
    for grad in gradients:
        for tensor, values in zip(tensors, grad):
            tensor.grad = None if values is None else torch.tensor(values, dtype=torch.float32)
        optim.step()
        entry = {}
        if kind == "sgd":
            entry["buffers"] = [
                None if optim.state.get(t, {}).get("momentum_buffer") is None
                else optim.state[t]["momentum_buffer"].tolist()
                for t in tensors
            ]
        else:
            entry["moments"] = [
                [optim.state[t]["exp_avg"].tolist(), optim.state[t]["exp_avg_sq"].tolist()]
                if t in optim.state else None
                for t in tensors
            ]
        entry["values"] = [t.detach().tolist() for t in tensors]
        steps.append(dict(gradients=grad, **entry))
    return dict(name=f"{kind}_groups", kind=kind, params=params, groups=groups, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "testdata/param_groups.json")
    args = parser.parse_args()
    shared_groups = [
        dict(name="decay", params=["weight", "bias"], lr=.1, weight_decay=.02),
        dict(name="no_decay", params=["gain"], lr=.05, weight_decay=0.),
    ]
    params = [dict(name="weight", value=[.5, -.25]), dict(name="bias", value=[.3, -.7]), dict(name="gain", value=[1., -1.])]
    gradients = [
        [[.3, -.1], [.2, .4], [.1, -.2]],
        [None, [.1, .1], None],
        [[0., 0.], None, [.3, .3]],
    ]
    cases = [
        build("sgd", params, shared_groups, gradients),
        # AdamW advances one shared step counter per optimizer call, so every
        # parameter needs a gradient each step to stay comparable with PyTorch's
        # per-parameter step state.
        build("adamw", params, shared_groups, [
            [[.3, -.1], [.2, .4], [.1, -.2]],
            [[.1, .1], [.1, .1], [.2, -.2]],
            [[0., 0.], [-.1, .1], [.3, .3]],
        ]),
    ]
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=cases), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(cases)} cases")


if __name__ == "__main__":
    main()
