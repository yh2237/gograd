"""Real-checkpoint token-sum duration predictor fixture; no weights are stored."""

import glob
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from sys import path as sys_path
sys_path.insert(0, os.environ.get("IRODORI_REFERENCE", r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"))
from irodori_tts.model import DurationPredictor

cache = Path(os.environ.get("HF_HOME", r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
checkpoint = glob.glob(str(cache / "hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*/model.safetensors"))[0]
model = DurationPredictor(text_dim=512, aux_dim=14, hidden_dim=1024, layers=3,
    dropout=0, speaker_dim=768, speaker_fusion="adarn_zero", caption_dim=512,
    caption_fusion="adarn_zero", caption_pooling="masked_mean", norm_eps=1e-5,
    architecture="token_sum_dual_adarn_zero_no_aux").eval()
with safe_open(checkpoint, framework="pt", device="cpu") as reader, torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(reader.get_tensor("duration_predictor." + name))
torch.set_num_threads(6)
text = torch.tensor([[[math.sin((i*512+j)*.013) for j in range(512)] for i in range(5)]], dtype=torch.float32)
speaker = torch.tensor([[[math.cos((i*768+j)*.007) for j in range(768)] for i in range(2)]], dtype=torch.float32)
caption = torch.tensor([[[math.sin((i*512+j)*.011) for j in range(512)] for i in range(3)]], dtype=torch.float32)
mask = torch.tensor([[True,True,True,False,False]])
cap_mask = torch.tensor([[True,True,False]])
cases = []
with torch.inference_mode():
    for has_speaker, has_caption in [(True,True),(False,False)]:
        start=time.perf_counter()
        result=model(text,text_mask=mask,aux_features=torch.zeros(1,14),speaker_state=speaker,
            speaker_mask=torch.ones(1,2,dtype=torch.bool),has_speaker=torch.tensor([has_speaker]),
            caption_state=caption,caption_mask=cap_mask,has_caption=torch.tensor([has_caption]))
        elapsed=(time.perf_counter()-start)*1000
        cases.append({"has_speaker":has_speaker,"has_caption":has_caption,"log_frames":result.item(),"elapsed_ms":elapsed})
        print(f"duration speaker={has_speaker} caption={has_caption}: {result.item():.8f}, {elapsed:.3f} ms")
Path("testdata/irodori_duration.json").write_text(json.dumps({"cases":cases},separators=(",",":"))+"\n",encoding="utf-8")
