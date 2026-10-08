"""Real-checkpoint 12-block DiT forward fixture using fixed encoded contexts."""

import glob
import dataclasses
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, os.environ.get("IRODORI_REFERENCE", r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"))
from irodori_tts.config import ModelConfig
from irodori_tts.model import DiffusionBlock, get_timestep_embedding, precompute_freqs_cis

cache=Path(os.environ.get("HF_HOME",r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
path=Path(glob.glob(str(cache/"hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*/model.safetensors"))[0])
with path.open("rb") as f:
    length=struct.unpack("<Q",f.read(8))[0]
    raw_cfg=json.loads(json.loads(f.read(length))["__metadata__"]["config_json"])
    allowed={field.name for field in dataclasses.fields(ModelConfig)}
    cfg=ModelConfig(**{k:v for k,v in raw_cfg.items() if k in allowed})
torch.set_num_threads(6)
cond=torch.nn.Sequential(torch.nn.Linear(512,1280,bias=False),torch.nn.SiLU(),torch.nn.Linear(1280,1280,bias=False),torch.nn.SiLU(),torch.nn.Linear(1280,3840,bias=False))
input_proj=torch.nn.Linear(32,1280)
blocks=torch.nn.ModuleList([DiffusionBlock(cfg) for _ in range(12)])
out_norm=torch.nn.Identity()  # replaced below with reference RMSNorm
from irodori_tts.model import RMSNorm
out_norm=RMSNorm(1280,eps=1e-5)
out_proj=torch.nn.Linear(1280,32)
modules={"cond_module":cond,"in_proj":input_proj,"blocks":blocks,"out_norm":out_norm,"out_proj":out_proj}
with safe_open(str(path),framework="pt",device="cpu") as reader,torch.no_grad():
    for prefix,module in modules.items():
        for name,param in module.named_parameters():
            param.copy_(reader.get_tensor(prefix+"."+name))
for module in modules.values(): module.eval()
x=torch.tensor([[[math.sin((i*32+j)*.17)*.4 for j in range(32)] for i in range(4)]],dtype=torch.float32)
text=torch.tensor([[[math.cos((i*512+j)*.009)*.3 for j in range(512)] for i in range(5)]],dtype=torch.float32)
speaker=torch.tensor([[[math.sin((i*768+j)*.006)*.2 for j in range(768)] for i in range(2)]],dtype=torch.float32)
caption=torch.tensor([[[math.cos((i*512+j)*.012)*.25 for j in range(512)] for i in range(3)]],dtype=torch.float32)
text_mask=torch.tensor([[True,True,True,False,False]])
speaker_mask=torch.tensor([[True,True]])
caption_mask=torch.tensor([[True,True,False]])
with torch.inference_mode():
    start=time.perf_counter()
    c=cond(get_timestep_embedding(torch.tensor([0.7]),512))[:,None,:]
    h=input_proj(x)
    freqs=precompute_freqs_cis(64,4)
    for block in blocks:
        h=block(h,c,text,text_mask,speaker,speaker_mask,caption,caption_mask,freqs)
    y=out_proj(out_norm(h))[0]
    elapsed=(time.perf_counter()-start)*1000
    print(f"DiT 12-block forward: {elapsed:.3f} ms")
Path("testdata/irodori_dit.json").write_text(json.dumps({"output":y.tolist(),"elapsed_ms":elapsed},separators=(",",":"))+"\n",encoding="utf-8")
