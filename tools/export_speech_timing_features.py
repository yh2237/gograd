"""Export a trusted UtauTTS features.pt cache to a Go-readable safetensors file.

Example: python tools/export_speech_timing_features.py --input
  D:/project/UtauTTS/out/speech-timing-target/features.pt --output
  %TEMP%/gograd-speech-timing/features.safetensors
The input is read only. torch.load uses pickle, so only use a trusted cache.
"""
import argparse
import json
import os
import random
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(Path(tempfile.gettempdir()) / "gograd-speech-timing" / "features.safetensors"))
    parser.add_argument("--split-seed", type=int, default=0, help="Python Random seed for the validation shuffle")
    args = parser.parse_args()
    src, dst = Path(args.input).resolve(), Path(args.output).resolve()
    if src == dst:
        parser.error("input and output must differ")
    data = torch.load(src, map_location="cpu", weights_only=False)
    order = list(range(len(data)))
    random.Random(args.split_seed).shuffle(order)
    header = {"__metadata__": {"format": "gograd-speech-timing-features-1", "count": str(len(data)),
                               "ids": json.dumps([item["id"] for item in data], ensure_ascii=False),
                               "split_seed": str(args.split_seed), "split_order": json.dumps(order)}}
    blobs = []
    offset = 0
    for i, item in enumerate(data):
        for key, dtype, code in (("ids", "<i8", "I64"), ("cont", "<f4", "F32"), ("target", "<f4", "F32")):
            values = np.ascontiguousarray(item[key], dtype=dtype)
            blob = values.tobytes()
            header[f"{i:05d}.{key}"] = {"dtype": code, "shape": list(values.shape),
                                        "data_offsets": [offset, offset + len(blob)]}
            blobs.append(blob)
            offset += len(blob)
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for blob in blobs:
            out.write(blob)
    print(f"{len(data)} utterances, {sum(len(x['ids']) for x in data)} frames -> {dst} ({os.path.getsize(dst)} bytes)")


if __name__ == "__main__":
    main()
