"""Real reference clip excerpt through torchaudio's default 44.1->48 kHz resampler."""
import json
import os
from pathlib import Path

import torchaudio

root=Path(os.environ.get("IRODORI_WAVS",r"D:\project\UtauTTS\.tmp-tsukuyomi\output\wavs"))
wav,sr=torchaudio.load(str(root/"VOICEACTRESS100_031.wav"))
assert sr==48000 and wav.shape[0]==1
source=torchaudio.functional.resample(wav[0,:12000],48000,44100)
want=torchaudio.functional.resample(source,44100,48000)
Path("testdata/irodori_resample_input.f32").write_bytes(source.numpy().tobytes())
Path("testdata/irodori_resample_output.f32").write_bytes(want.numpy().tobytes())
Path("testdata/irodori_resample.json").write_text(json.dumps({"input_rate":44100,"output_rate":48000,"input_samples":source.numel(),"output_samples":want.numel()},separators=(",",":"))+"\n",encoding="utf-8")
