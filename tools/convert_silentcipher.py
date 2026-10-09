"""Convert silentcipher's .ckpt weights to a safetensors file for Go.

The HF snapshot ships PyTorch zip checkpoints with a module. prefix, which
gograd's reader cannot use. This writes only the tensors the encode path needs
-- the carrier encoder and carrier decoder -- and drops the decoder that the
reference never calls and the optimizer state.

    python tools/convert_silentcipher.py SOURCE_CHECKPOINT_DIR DEST.safetensors

DEST must be outside this repository: the weights are a third-party model, not
project fixtures.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# enc_c encodes the carrier; dec_c predicts the magnitude perturbation. dec_m
# decodes messages and is never called on the encode path, so it is skipped.
REQUIRED = ["enc_c.ckpt", "dec_c.ckpt"]


def convert(source, dest) -> None:
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")
    tensors = {}
    for name in REQUIRED:
        # Both files use the same main.N layout, so each one's tensors are
        # namespaced by the file stem the reference loads them from.
        prefix = name.rsplit(".", 1)[0] + "."
        checkpoint = torch.load(source / name, map_location="cpu", weights_only=True)
        for key, value in checkpoint.items():
            if not key.startswith("module."):
                raise SystemExit(f"{name}: unexpected key {key!r}, expected a module. prefix")
            key = prefix + key[len("module."):]
            if key.endswith("num_batches_tracked"):
                # A batch counter, not a weight; the reference reads the shape
                # of the running statistics instead.
                continue
            if value.dtype != torch.float32:
                raise SystemExit(f"{name}/{key}: {value.dtype} is not float32")
            tensors[key] = value.contiguous()
    header = {
        "model": "silentcipher",
        "model_type": "44.1k",
        "sample_rate": 44100,
        "n_fft": 4096,
        "hop_length": 2048,
        "message_dim": 5,
        "message_len": 21,
        "message_band_size": 1024,
        "message_sdr": 47,
        "average_energy": 0.002837200844477648,
        "source": [str(source / name) for name in REQUIRED],
    }
    hparams = source / "hparams.yaml"
    if hparams.is_file():
        header["hparams"] = hparams.read_text(encoding="utf-8")
    save_file(tensors, str(dest), metadata={k: json.dumps(v) for k, v in header.items()})
    total = sum(t.numel() for t in tensors.values())
    print(f"wrote {dest}: {len(tensors)} tensors, {total} parameters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory holding enc_c.ckpt and dec_c.ckpt")
    parser.add_argument("dest", type=Path, help="output safetensors file, outside the repository")
    args = parser.parse_args()
    try:
        repo = Path(__file__).resolve().parent.parent
    except NameError:
        repo = Path(".").resolve()
    if args.dest.resolve().is_relative_to(repo):
        raise SystemExit(f"destination {args.dest} must be outside {repo}")
    convert(args.source, args.dest)


if __name__ == "__main__":
    main()
