"""Print safetensors header summary without loading model weights."""
import collections
import json
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as f:
    length = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(length))
types = collections.Counter(v["dtype"] for k, v in header.items() if k != "__metadata__")
print("tensor_count:", sum(types.values()))
print("dtypes:", dict(types))
print("metadata:", header.get("__metadata__"))
print("first_names:", list(k for k in header if k != "__metadata__")[:25])
