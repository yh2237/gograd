"""Generate PyTorch loss values and gradients for Go-only parity tests."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "testdata/losses.json")
    args = parser.parse_args()
    prediction = [[1.5, -2.0, 0.25], [3.5, -0.5, -10.0]]
    target = [[0.0, 0.5, 1.0], [1.0, 0.0, 0.75]]
    weights = [[1.0, 0.0, 2.0], [0.5, 1.0, 3.0]]
    functions = {
        "mse": F.mse_loss,
        "l1": F.l1_loss,
        "bce_logits": F.binary_cross_entropy_with_logits,
    }
    cases = []
    for kind, function in functions.items():
        for reduction in ("mean", "sum"):
            for use_weight in (False, True):
                prediction_tensor = torch.tensor(prediction, requires_grad=True)
                target_tensor = torch.tensor(target)
                weight_tensor = torch.tensor(weights) if use_weight else None
                loss = function(prediction_tensor, target_tensor, weight=weight_tensor, reduction=reduction)
                (grad,) = torch.autograd.grad(loss, prediction_tensor)
                cases.append(dict(
                    kind=kind, reduction=reduction, weight=use_weight,
                    prediction=prediction, target=target,
                    weights=weights if use_weight else None,
                    loss=float(loss.detach()), grad=grad.tolist(),
                ))
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=cases), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(cases)} cases")


if __name__ == "__main__":
    main()
