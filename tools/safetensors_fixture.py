"""Create or verify a small PyTorch safetensors checkpoint."""
import sys
import torch
from safetensors.torch import load_file, save_file

mode, path = sys.argv[1:3]
expected = {
    "block.weight": torch.tensor([[1.25, -2.5], [3.75, 4.5]], dtype=torch.float32),
    "block.bias": torch.tensor([-0.125, 0.375], dtype=torch.float32),
}
if mode == "write":
    save_file(expected, path)
elif mode == "verify":
    got = load_file(path)
    assert set(got) == set(expected)
    for key in expected:
        assert torch.equal(got[key], expected[key]), key
else:
    raise ValueError(mode)
