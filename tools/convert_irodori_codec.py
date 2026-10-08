"""Convert the trusted cached DACVAE .pth to F32 safetensors for Go.

Usage: python tools/convert_irodori_codec.py C:\\...\\weights.pth C:\\...\\codec.safetensors
The output contains checkpoint weights; keep it outside the repository.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("destination must be outside the repository (weights are not fixtures)")
    # This source is the trusted, locally cached HF checkpoint. torch.load of
    # arbitrary .pth files is unsafe; conversion is deliberately opt-in.
    raw = torch.load(args.source, map_location="cpu", weights_only=False)
    state = raw["state_dict"]
    converted = {}
    for name, tensor in state.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            stem = name[:-len("weight_v")]
            g = state[stem + "weight_g"].float()
            v = tensor.float()
            dims = tuple(range(1, v.ndim)) if stem.startswith("encoder.") or stem.startswith("quantizer.") else None
            # ConvTranspose weight norm is along dim=1, unlike Conv1d dim=0.
            if stem.startswith("decoder.") and v.ndim == 3 and g.shape[1] == v.shape[1]:
                dims = (0, 2)
            if dims is None:
                dims = tuple(range(1, v.ndim))
            converted[stem + "weight"] = (v * (g / torch.linalg.vector_norm(v, dim=dims, keepdim=True))).contiguous()
        else:
            converted[name] = tensor.float().contiguous()
    metadata = {"format": "gograd-dacvae-f32-v1", "kwargs_json": json.dumps(raw["metadata"]["kwargs"])}
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(args.destination), metadata=metadata)
    print(f"saved {len(converted)} F32 tensors to {args.destination}")


if __name__ == "__main__":
    main()
