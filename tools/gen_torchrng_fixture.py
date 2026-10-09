"""PyTorch CPU generator reference for gograd/torchrng.

Writes the raw byte streams of torch.randn and torch.rand plus the engine words
so the Go implementation can be checked bit for bit without Python at test
time. Requires torch; run from the repository root.
"""

import json
from pathlib import Path

import torch

SEEDS = [0, 1, 2, 7231, 5489, 4294967295, 67280421310721, 12345678901234567]
# Lengths around and beyond the 16-element block boundary: the short serial
# path, exact multiples, and partial tails that force a recomputed final block.
LENGTHS = [1, 2, 7, 15, 16, 17, 116 * 32, 3712, 4 * 32, 100, 16*5 + 3]

blob = bytearray()
streams = []
words = {}
for seed in SEEDS:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    words[str(seed)] = [int(v) for v in torch.randint(
        -(2**31), 2**31 - 1, (8,), dtype=torch.int64, generator=generator).tolist()]
    for n in LENGTHS:
        entries = []
        for dtype in (torch.float32, torch.float64):
            gen = torch.Generator(device="cpu").manual_seed(seed)
            entries.append(torch.randn(n, generator=gen, dtype=dtype).numpy().tobytes())
            gen = torch.Generator(device="cpu").manual_seed(seed)
            entries.append(torch.rand(n, generator=gen, dtype=dtype).numpy().tobytes())
        start = len(blob)
        blob.extend(b"".join(entries))
        streams.append({"seed": seed, "n": n, "offset": start, "bytes": len(blob) - start})

Path("testdata/torchrng_streams.bin").write_bytes(bytes(blob))
Path("testdata/torchrng_streams.json").write_text(
    json.dumps({"streams": streams, "words": words}, separators=(",", ":")) + "\n", encoding="utf-8")
print(f"{len(streams)} streams, {len(blob)} bytes")
