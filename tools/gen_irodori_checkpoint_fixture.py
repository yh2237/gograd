"""Dump one small v4.1-Small state tensor for streaming-load parity."""
import json
import sys
from pathlib import Path

from safetensors import safe_open

name = "blocks.0.attention.k_norm.weight"
with safe_open(sys.argv[1], framework="pt", device="cpu") as checkpoint:
    tensor = checkpoint.get_tensor(name)
    fixture = {"name": name, "shape": list(tensor.shape), "values": tensor.flatten().tolist(),
               "tensor_count": len(checkpoint.keys())}
Path("testdata/irodori_checkpoint_tensor.json").write_text(
    json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
)
