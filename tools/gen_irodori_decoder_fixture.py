"""Real DACVAE four-frame waveform fixture (no checkpoint weights)."""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

root=Path(os.environ.get("IRODORI_REFERENCE",r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"))
sys.path.insert(0,str(root))
from irodori_tts.codec import DACVAECodec
cache=Path(os.environ.get("HF_HOME",r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
codec_path=Path(glob.glob(str(cache/"hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/*/weights.pth"))[0])
torch.set_num_threads(6)
codec=DACVAECodec.load(repo_id=str(codec_path),device="cpu",dtype=torch.float32,normalize_db=-16.0)
latent=np.fromfile("testdata/irodori_reference_latent.f32",dtype="<f4").reshape(-1,32)[:4]
z=torch.from_numpy(latent.T.copy())[None]
with torch.inference_mode():
    start=time.perf_counter();wav=codec.model.decode(z)[0,0].cpu().contiguous();elapsed=(time.perf_counter()-start)*1000
Path("testdata/irodori_decoder_wave.f32").write_bytes(wav.numpy().tobytes())
Path("testdata/irodori_decoder.json").write_text(json.dumps({"frames":4,"samples":wav.numel(),"first":wav[:16].tolist(),"last":wav[-16:].tolist(),"elapsed_ms":elapsed},separators=(",",":"))+"\n",encoding="utf-8")
print(f"decoded {wav.numel()} samples in {elapsed:.3f} ms")
