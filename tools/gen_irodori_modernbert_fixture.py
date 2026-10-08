"""Real-checkpoint ModernBERT-ja output slices for the Go inference port."""

import glob
import json
import os
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

cache = Path(os.environ.get("HF_HOME", r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
path = Path(glob.glob(str(cache / "hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*/model.safetensors"))[0])
with path.open("rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    metadata = json.loads(f.read(n))["__metadata__"]
cfg = json.loads(metadata["text_encoder_config_json"])
cfg.pop("model_type")
torch.set_num_threads(6)
model = AutoModel.from_config(AutoConfig.for_model("modernbert", **cfg)).eval()
prefix = "pretrained_text_backbone.backbone."
with safe_open(str(path), framework="pt", device="cpu") as reader, torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(reader.get_tensor(prefix + name))

def project(reader, state, mask, kind):
    import torch.nn.functional as F
    base = F.linear(state, reader.get_tensor(kind+"_encoder.projector.weight"), reader.get_tensor(kind+"_encoder.projector.bias"))
    weight = reader.get_tensor(kind+"_encoder.residual_norm.weight")
    normed = state * torch.rsqrt((state*state).mean(dim=-1,keepdim=True)+1e-5)*weight
    up = F.linear(normed, reader.get_tensor(kind+"_encoder.residual_up.weight"), reader.get_tensor(kind+"_encoder.residual_up.bias"))
    residual = F.linear(F.silu(up), reader.get_tensor(kind+"_encoder.residual_down.weight"), reader.get_tensor(kind+"_encoder.residual_down.bias"))
    x = (base+residual)*mask[...,None]
    w = reader.get_tensor(kind+"_norm.weight")
    return x*torch.rsqrt((x*x).mean(dim=-1,keepdim=True)+1e-5)*w

cases = json.loads(Path("testdata/irodori_tokenizer.json").read_text(encoding="utf-8"))
selected = [(12, 16), (21, 32), (64, 160)]
output = []
with torch.inference_mode():
    for index, length in selected:
        ids = cases[index]["padded_256"][:length]
        mask = [x != 3 for x in ids]
        x = torch.tensor([ids], dtype=torch.long)
        m = torch.tensor([mask], dtype=torch.bool)
        start = time.perf_counter()
        y = model(input_ids=x, attention_mask=m, return_dict=True).last_hidden_state[0]
        elapsed = time.perf_counter() - start
        y = y * m[0, :, None]
        rows = sorted(set([0, min(len(ids)-1, sum(mask)//2), len(ids)-1]))
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            text_projection=project(reader,y,m[0],"text")
            caption_projection=project(reader,y,m[0],"caption")
        output.append({
            "ids": ids, "mask": mask, "rows": rows,
            "slices": [y[r, :16].tolist() for r in rows],
            "row_sums": y.sum(dim=1).tolist(),
            "text_slices": [text_projection[r,:16].tolist() for r in rows],
            "caption_slices": [caption_projection[r,:16].tolist() for r in rows],
            "elapsed_ms": round(elapsed * 1000, 3),
        })
        print(f"ModernBERT case {index}, length {length}: {elapsed*1000:.3f} ms")
Path("testdata/irodori_modernbert.json").write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
