"""Real-checkpoint raw-text and eight-reference integrated denoiser fixture."""

import dataclasses
import glob
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

root=Path(os.environ.get("IRODORI_REFERENCE",r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"))
sys.path.insert(0,str(root))
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.text_normalization import normalize_text
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.rf import sample_euler_rf_cfg

cache=Path(os.environ.get("HF_HOME",r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
snapshot=Path(glob.glob(str(cache/"hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*"))[0])
checkpoint=snapshot/"model.safetensors"
with checkpoint.open("rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]
    metadata=json.loads(f.read(n))["__metadata__"]
raw_cfg=json.loads(metadata["config_json"])
cfg=ModelConfig(**{k:v for k,v in raw_cfg.items() if k in {f.name for f in dataclasses.fields(ModelConfig)}})
text_cfg=json.loads(metadata["text_encoder_config_json"])
torch.set_num_threads(6)
model=TextToLatentRFDiT(cfg,pretrained_backbone_config=text_cfg,load_pretrained_backbone_weights=False).eval()
with safe_open(str(checkpoint),framework="pt",device="cpu") as reader,torch.no_grad():
    for name,param in model.named_parameters():
        param.copy_(reader.get_tensor(name))
tokenizer=PretrainedTextTokenizer.from_pretrained(repo_id=str(snapshot/"tokenizer"),add_bos=cfg.text_add_bos,local_files_only=True)
text="こんにちは、今日もいい天気ですね。"
caption="やさしい声で話す女性"
text_norm=normalize_text(text)
caption_norm=normalize_text(caption)
ids,mask=tokenizer.batch_encode([text_norm],max_length=24)
cap_ids,cap_mask=tokenizer.batch_encode([caption_norm],max_length=24)
latent=np.fromfile("testdata/irodori_reference_latent.f32",dtype="<f4").reshape(1,-1,32)
ref=torch.from_numpy(latent.copy())
ref_mask=torch.ones(ref.shape[:2],dtype=torch.bool)
x=torch.tensor([[[math.sin((i*32+j)*.17)*.4 for j in range(32)] for i in range(4)]],dtype=torch.float32)
def summary(x):
    v=x.reshape(-1)
    return {"shape":list(x.shape),"first":v[:16].tolist(),"last":v[-16:].tolist(),"sum":v.sum().item(),"square_sum":(v*v).sum().item()}
with torch.inference_mode():
    start=time.perf_counter()
    ts,tm,ss,sm,cs,cm=model.encode_conditions(ids,mask,ref,ref_mask,cap_ids,cap_mask)
    condition_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    y=model.forward_with_encoded_conditions(x,torch.tensor([.7]),ts,tm,ss,sm,cs,cm)
    forward_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    duration=model.predict_duration_log_frames(text_state=ts,text_mask=tm,speaker_state=ss,speaker_mask=sm,duration_features=torch.zeros(1,cfg.duration_aux_dim),has_speaker=torch.tensor([True]),caption_state=cs,caption_mask=cm,has_caption=torch.tensor([True]))
    duration_ms=(time.perf_counter()-start)*1000
    seed=7231
    noise=torch.randn((1,4,32),generator=torch.Generator(device="cpu").manual_seed(seed),dtype=torch.float32)
    Path("testdata/irodori_sampler_noise.f32").write_bytes(noise.numpy().tobytes())
    start=time.perf_counter()
    sampled=sample_euler_rf_cfg(model,ids,mask,ref,ref_mask,sequence_length=4,caption_input_ids=cap_ids,caption_mask=cap_mask,num_steps=3,cfg_scale_text=3.0,cfg_scale_speaker=5.0,cfg_scale_caption=3.0,cfg_guidance_mode="independent",cfg_min_t=.5,cfg_max_t=1.0,seed=seed,t_schedule_mode="sway",sway_coeff=-1.0,use_context_kv_cache=False)
    sampler_ms=(time.perf_counter()-start)*1000
    cfg_modes={}
    for mode,scales in [("joint",(3.0,3.0,3.0)),("alternating",(3.0,5.0,3.0))]:
        start=time.perf_counter()
        result_mode=sample_euler_rf_cfg(model,ids,mask,ref,ref_mask,sequence_length=4,caption_input_ids=cap_ids,caption_mask=cap_mask,num_steps=3,cfg_scale_text=scales[0],cfg_scale_speaker=scales[1],cfg_scale_caption=scales[2],cfg_guidance_mode=mode,cfg_min_t=.5,cfg_max_t=1.0,seed=seed,t_schedule_mode="sway",sway_coeff=-1.0,use_context_kv_cache=False)
        cfg_modes[mode]={"final":summary(result_mode),"elapsed_ms":(time.perf_counter()-start)*1000}
result={"text":text,"normalized_text":text_norm,"caption":caption,"normalized_caption":caption_norm,"ids":ids[0].tolist(),"mask":mask[0].tolist(),"caption_ids":cap_ids[0].tolist(),"caption_mask":cap_mask[0].tolist(),"reference_frames":int(ref.shape[1]),"text_state":summary(ts),"speaker_state":summary(ss),"caption_state":summary(cs),"denoiser":summary(y),"duration":duration.item(),"condition_ms":condition_ms,"forward_ms":forward_ms,"duration_ms":duration_ms,"sampler":{"seed":seed,"noise":summary(noise),"final":summary(sampled),"elapsed_ms":sampler_ms,"modes":cfg_modes}}
Path("testdata/irodori_integrated.json").write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
print(f"condition {condition_ms:.3f} ms, denoiser {forward_ms:.3f} ms, duration {duration_ms:.3f} ms")
print(f"sampler {sampler_ms:.3f} ms")

if os.environ.get("IRODORI_FULL_BENCH") == "1":
    from irodori_tts.codec import DACVAECodec
    from irodori_tts.inference_runtime import _load_audio
    codec_path=Path(glob.glob(str(cache/"hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/*/weights.pth"))[0])
    codec=DACVAECodec.load(repo_id=str(codec_path),device="cpu",dtype=torch.float32,normalize_db=-16.0)
    wav_root=Path(os.environ.get("IRODORI_WAVS",r"D:\project\UtauTTS\.tmp-tsukuyomi\output\wavs"))
    start=time.perf_counter()
    pieces=[]
    for n in range(31,39):
        wav,sr=_load_audio(wav_root/f"VOICEACTRESS100_{n:03d}.wav")
        pieces.append(codec.encode_waveform(wav[None],sr,normalize_db=-16.0,ensure_max=True))
    reference=torch.cat(pieces,dim=1)
    codec_ms=(time.perf_counter()-start)*1000
    latent_steps=round(math.expm1(duration.item()))
    seed=7231
    noise=torch.randn((1,latent_steps,32),generator=torch.Generator(device="cpu").manual_seed(seed),dtype=torch.float32)
    Path("testdata/irodori_full_noise.f32").write_bytes(noise.numpy().tobytes())
    start=time.perf_counter()
    final=sample_euler_rf_cfg(model,ids,mask,reference,torch.ones(reference.shape[:2],dtype=torch.bool),sequence_length=latent_steps,caption_input_ids=cap_ids,caption_mask=cap_mask,num_steps=1,cfg_scale_text=3.0,cfg_scale_speaker=5.0,cfg_scale_caption=3.0,cfg_guidance_mode="independent",cfg_min_t=.5,cfg_max_t=1.0,seed=seed,t_schedule_mode="linear",use_context_kv_cache=False)
    sample_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();audio=codec.decode_latent(final)[0,0].cpu();decode_ms=(time.perf_counter()-start)*1000
    indices=list(range(0,audio.numel(),max(1,audio.numel()//256)))
    full={"latent_steps":latent_steps,"sample_rate":48000,"codec_ms":codec_ms,"sample_ms":sample_ms,"decode_ms":decode_ms,"final_latent":summary(final),"wave_samples":audio.numel(),"wave_indices":indices,"wave_values":audio[indices].tolist()}
    Path("testdata/irodori_full_one_step.json").write_text(json.dumps(full,separators=(",",":"))+"\n",encoding="utf-8")
    print(f"full one-step sentence: codec {codec_ms:.1f} ms, sample {sample_ms:.1f} ms, decode {decode_ms:.1f} ms, {audio.numel()} samples")

if os.environ.get("IRODORI_DIT_CUDA_BENCH") == "1":
    # Only the DiT moves to CUDA; the F32 full checkpoint would exceed 3 GB.
    for part in [model.cond_module,model.in_proj,model.blocks,model.out_norm,model.out_proj]:part.to("cuda")
    noise=np.fromfile("testdata/irodori_full_noise.f32",dtype="<f4").reshape(1,-1,32)
    x=torch.from_numpy(noise.copy()).to("cuda")
    tc,sc,cc=ts.to("cuda"),ss.to("cuda"),cs.to("cuda")
    tm,sm,cm=tm.to("cuda"),sm.to("cuda"),cm.to("cuda")
    def cat_four(value,zero):return torch.cat([value,zero,value,value],dim=0)
    text_b=cat_four(tc,torch.zeros_like(tc))
    text_m=cat_four(tm,torch.zeros_like(tm))
    speaker_b=torch.cat([sc,sc,torch.zeros_like(sc),sc],dim=0)
    speaker_m=torch.cat([sm,sm,torch.zeros_like(sm),sm],dim=0)
    caption_b=torch.cat([cc,cc,cc,torch.zeros_like(cc)],dim=0)
    caption_m=torch.cat([cm,cm,cm,torch.zeros_like(cm)],dim=0)
    with torch.inference_mode():
        torch.cuda.synchronize();start=time.perf_counter()
        v=model.forward_with_encoded_conditions(x.repeat(4,1,1),torch.full((4,),.999,device="cuda"),text_b,text_m,speaker_b,speaker_m,caption_b,caption_m)
        a,b,c,d=v.chunk(4,dim=0)
        final=x+(a+3*(a-b)+5*(a-c)+3*(a-d))*(-.999)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
    fixture=json.loads(Path("testdata/irodori_full_one_step.json").read_text())
    torch_final=torch.tensor(fixture["final_latent"]["first"])
    error=(final.cpu().reshape(-1)[:16]-torch_final).abs().max().item()
    print(f"PyTorch CUDA DiT one-step 116 frames: {elapsed:.3f} ms, peak_allocated={torch.cuda.max_memory_allocated()} bytes, fixture_first_max_abs={error:.9g}")

if os.environ.get("IRODORI_CFG_AUDIT") == "1":
    zero=lambda x:torch.zeros_like(x)
    bundles=[(ts,tm,ss,sm,cs,cm),(zero(ts),zero(tm),ss,sm,cs,cm),(ts,tm,zero(ss),zero(sm),cs,cm),(ts,tm,ss,sm,zero(cs),zero(cm))]
    t=torch.tensor([.999],dtype=torch.float32)
    with torch.inference_mode():
        serial=[model.forward_with_encoded_conditions(noise,t,*bundle)[0] for bundle in bundles]
        batch_args=[torch.cat([bundle[i] for bundle in bundles],dim=0) for i in range(6)]
        batched=model.forward_with_encoded_conditions(noise.repeat(4,1,1),t.repeat(4),*batch_args)
        errors=[(batched[i]-serial[i]).abs().max().item() for i in range(4)]
    print("independent branch batch-versus-serial max_abs",errors)
    u=torch.linspace(0.0,1.0,4);u=u-(torch.cos(.5*math.pi*u)+u-1.0);grid=(1-u)*.999
    print("independent grid",grid.tolist())
    traces={}
    # Independent: all CFG branches in one batched forward.
    xt=noise.clone();trace=[]
    with torch.inference_mode():
        for step in range(3):
            time_val=grid[step];next_time=grid[step+1]
            batched_args=[torch.cat([bundle[i] for bundle in bundles],dim=0) for i in range(6)]
            branches=model.forward_with_encoded_conditions(xt.repeat(4,1,1),time_val.expand(4),*batched_args).chunk(4,dim=0)
            a,b,c,d=branches;v=a+3*(a-b)+5*(a-c)+3*(a-d) if .5<=time_val.item()<=1.0 else a
            xt=xt+v*(next_time-time_val)
            trace.append({"t":time_val.item(),"next":next_time.item(),"velocity":v[0].tolist(),"x":xt[0].tolist()})
    print("independent trace final vs fixture",(xt-sampled).abs().max().item())
    traces["independent"]=trace
    # Joint and alternating: the reference evaluates one conditional and one
    # unconditional forward per step with batch 1, exactly like gograd does.
    for mode,spk in [("joint",3.0),("alternating",5.0)]:
        xt=noise.clone();trace=[]
        for step in range(3):
            time_val=grid[step];next_time=grid[step+1]
            with torch.inference_mode():
                cond=model.forward_with_encoded_conditions(xt,time_val.reshape(1),ts,tm,ss,sm,cs,cm)[0]
                if .5<=time_val.item()<=1.0:
                    if mode=="joint":
                        branch=(zero(ts),zero(tm),zero(ss),zero(sm),zero(cs),zero(cm))
                    else:
                        name=("text","speaker","caption")[step%3]
                        branch=(zero(ts),zero(tm),ss,sm,cs,cm) if name=="text" else (ts,tm,zero(ss),zero(sm),cs,cm) if name=="speaker" else (ts,tm,ss,sm,zero(cs),zero(cm))
                    unc=model.forward_with_encoded_conditions(xt,time_val.reshape(1),*branch)[0]
                    v=cond+(spk if mode=="joint" else {"text":3.0,"speaker":5.0,"caption":3.0}[("text","speaker","caption")[step%3]])*(cond-unc)
                else:
                    v=cond
                xt=xt+v*(next_time-time_val)
            trace.append({"t":time_val.item(),"next":next_time.item(),"velocity":v.tolist(),"x":xt[0].tolist()})
        traces[mode]=trace
    Path("testdata/irodori_cfg_branches.json").write_text(json.dumps({"t":.999,"batch_vs_serial_max_abs":errors,"batched":[v.tolist() for v in batched],"serial":[v.tolist() for v in serial],"trace":trace,"traces":traces},separators=(",",":"))+"\n",encoding="utf-8")
