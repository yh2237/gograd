"""Small real-reference WAV, codec latent, and speaker-state parity fixture."""

import dataclasses
import glob
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torchaudio
from safetensors import safe_open

root=Path(os.environ.get("IRODORI_REFERENCE",r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"))
sys.path.insert(0,str(root))
from irodori_tts.codec import DACVAECodec
from irodori_tts.config import ModelConfig
from irodori_tts.model import RMSNorm, ReferenceLatentEncoder, patch_sequence_with_mask
from irodori_tts.inference_runtime import _load_audio

cache=Path(os.environ.get("HF_HOME",r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
codec_path=Path(glob.glob(str(cache/"hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/*/weights.pth"))[0])
checkpoint=Path(glob.glob(str(cache/"hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*/model.safetensors"))[0])
with checkpoint.open("rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]
    raw=json.loads(json.loads(f.read(n))["__metadata__"]["config_json"])
allowed={x.name for x in dataclasses.fields(ModelConfig)}
cfg=ModelConfig(**{k:v for k,v in raw.items() if k in allowed})
torch.set_num_threads(6)
codec=DACVAECodec.load(repo_id=str(codec_path),device="cpu",dtype=torch.float32,normalize_db=-16.0)
speaker=ReferenceLatentEncoder(cfg).eval()
speaker_norm=RMSNorm(cfg.speaker_dim,eps=cfg.norm_eps).eval()
with safe_open(str(checkpoint),framework="pt",device="cpu") as reader,torch.no_grad():
    for name,param in speaker.named_parameters(): param.copy_(reader.get_tensor("speaker_encoder."+name))
    for name,param in speaker_norm.named_parameters(): param.copy_(reader.get_tensor("speaker_norm."+name))

def summary(tensor):
    t=tensor.float().reshape(-1)
    return {"shape":list(tensor.shape),"first":t[:16].tolist(),"last":t[-16:].tolist(),
            "middle":t[len(t)//2:len(t)//2+16].tolist(),"max_abs":t.abs().max().item(),
            "sum":t.sum().item(),"square_sum":torch.square(t).sum().item()}

wavs=Path(os.environ.get("IRODORI_WAVS",r"D:\project\UtauTTS\.tmp-tsukuyomi\output\wavs"))
records=[]; pieces=[]
with torch.inference_mode():
    for number in range(31,39):
        path=wavs/f"VOICEACTRESS100_{number:03d}.wav"
        wav,sr=_load_audio(path)
        start=time.perf_counter()
        mono=wav.float().mean(dim=0) if wav.shape[0]>1 else wav.float()[0]
        resampled=torchaudio.functional.resample(mono[None,None],sr,codec.sample_rate)[0,0] if sr!=codec.sample_rate else mono
        normalized=codec._normalize_loudness(resampled,codec.sample_rate,-16.0)
        latent=codec.encode_waveform(wav.unsqueeze(0),sr,normalize_db=-16.0,ensure_max=True).cpu()[0]
        elapsed=(time.perf_counter()-start)*1000
        pieces.append(latent)
        records.append({"file":path.name,"sample_rate":sr,"samples":wav.shape[1],
                        "resampled":summary(resampled),"normalized":summary(normalized),
                        "latent":summary(latent),"elapsed_ms":elapsed})
        print(f"{path.name}: {sr} Hz, {latent.shape[0]} latent frames, {elapsed:.1f} ms",flush=True)
    reference=torch.cat(pieces,dim=0)[None]
    # Audio latents are test inputs, not checkpoint weights (about 147 KiB).
    (Path("testdata")/"irodori_reference_latent.f32").write_bytes(reference.contiguous().numpy().tobytes())
    mask=torch.ones(reference.shape[:2],dtype=torch.bool)
    patched,patched_mask=patch_sequence_with_mask(reference,mask,cfg.speaker_patch_size)
    start=time.perf_counter()
    state=speaker_norm(speaker(patched,patched_mask))[0]
    mean=state.mean(dim=0,keepdim=True)
    state=torch.cat([mean,state],dim=0)
    elapsed=(time.perf_counter()-start)*1000
    print(f"combined speaker state {list(state.shape)}: {elapsed:.1f} ms",flush=True)
result={"clips":records,"combined_latent":summary(reference),"speaker_state":summary(state),
        "speaker_first_rows":state[:2,:16].tolist(),"speaker_last_rows":state[-2:,:16].tolist(),
        "speaker_elapsed_ms":elapsed,"codec_sample_rate":codec.sample_rate,"codec_hop_length":int(codec.model.hop_length)}
Path("testdata/irodori_reference.json").write_text(json.dumps(result,separators=(",",":"))+"\n",encoding="utf-8")
