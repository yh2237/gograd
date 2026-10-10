"""Generate PyTorch SGD updates/buffers for Go-only CPU/CUDA tests."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "testdata/sgd.json")
    args = parser.parse_args()
    configs = [
        ("plain", {}), ("coupled_decay", dict(weight_decay=.2)),
        ("momentum", dict(momentum=.9)), ("dampening", dict(momentum=.8, dampening=.2, weight_decay=.1)),
        ("nesterov", dict(momentum=.9, nesterov=True, weight_decay=.1)),
        ("maximize", dict(momentum=.9, maximize=True, weight_decay=.1)),
        ("no_momentum_dampening", dict(dampening=.4)),
    ]
    initial = [[.5, -.25, 1.5], [.3, -.7]]
    gradients = [
        [[.3, -.1, .2], None], [None, [-.2, .4]], [[0., 0., 0.], None],
        [[-.2, .4, -.3], [.1, -.1]], [None, None], [[.1, .1, -.1], [-.3, .2]],
    ]
    cases = []
    for name, config in configs:
        params = [torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)) for v in initial]
        optimizer = torch.optim.SGD(params, lr=.1, **config)
        steps = []
        for i, grad in enumerate(gradients):
            lr = 0. if i == 2 else .03 if i >= 4 else .1
            optimizer.param_groups[0]["lr"] = lr
            for param, values in zip(params, grad):
                param.grad = None if values is None else torch.tensor(values, dtype=torch.float32)
            optimizer.step()
            buffers = [optimizer.state.get(p, {}).get("momentum_buffer") for p in params]
            steps.append(dict(lr=lr, gradients=grad, values=[p.detach().tolist() for p in params],
                              buffers=[None if b is None else b.tolist() for b in buffers]))
        options = dict(LR=.1, Momentum=config.get("momentum", 0.), Dampening=config.get("dampening", 0.),
                       WeightDecay=config.get("weight_decay", 0.), Nesterov=config.get("nesterov", False), Maximize=config.get("maximize", False))
        cases.append(dict(name=name, options=options, initial=initial, steps=steps))
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=cases), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(cases)} cases")


if __name__ == "__main__":
    main()
