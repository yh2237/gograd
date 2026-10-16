import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1] / "torchpth" / "testdata"
root.mkdir(parents=True, exist_ok=True)
x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
checkpoint = {
    "bert": {"plain": x[:, 1:4], "transposed": x.T},
    "predictor": {"half": x[:, ::2].half(), "bfloat": x[:, 2:].to(torch.bfloat16)},
}
torch.save(checkpoint, root / "tiny.pth")
expected = {f"{group}.{name}": {"shape": list(value.shape), "data": value.float().contiguous().flatten().tolist()}
            for group, params in checkpoint.items() for name, value in params.items()}
(root / "tiny.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
