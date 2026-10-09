"""Reference STFT/iSTFT and Resample shapes for gograd/dsp.

Mirrors the silentcipher STFT singleton, torch.stft/torch.istft defaults and
torchaudio.functional.resample, so the Go port can be checked without Python at
test time.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch
import torchaudio

CASES = [4096, 4096 + 1, 4096 * 2, 132300, 48000 * 3, 4096 * 3 - 7]
N_FFT, HOP = 4096, 2048

def torchaudio_resample(x, orig, new):
    return torchaudio.functional.resample(torch.from_numpy(x).view(1, -1), orig_freq=orig, new_freq=new).squeeze().numpy()
N_FFT, HOP = 4096, 2048


def stft_singleton(x):
    x = torch.nn.functional.pad(x, (0, N_FFT - x.shape[1] % N_FFT))
    fft = torch.stft(x, N_FFT, HOP, N_FFT, window=torch.hann_window(N_FFT),
                     return_complex=True)
    real, imag = fft.real, fft.imag
    squared = real**2 + imag**2
    eps = torch.ones_like(squared) * (squared == 0).float() * 1e-24
    magnitude = torch.sqrt(squared + eps) - torch.sqrt(eps)
    phase = torch.atan2(imag, real)
    return magnitude, phase


def inverse(magnitude, phase, num_samples):
    recombine = magnitude * torch.cos(phase) + 1j * magnitude * torch.sin(phase)
    out = torch.istft(recombine, N_FFT, HOP, N_FFT, window=torch.hann_window(N_FFT)).unsqueeze(1)
    padding = N_FFT - (num_samples % N_FFT)
    return out[:, :, :-padding]


rng = np.random.default_rng(5)
blob = bytearray()
index = []
for n in CASES:
    signal = (np.sin(np.arange(n) * 0.015) * 0.4 + rng.standard_normal(n) * 0.02).astype(np.float32)
    t = torch.from_numpy(signal.copy()).reshape(1, -1)
    magnitude, phase = stft_singleton(t)
    frames = magnitude.shape[-1]
    num_samples = t.shape[1]
    rebuilt = inverse(magnitude, phase, num_samples)
    wav = rebuilt[0, 0].numpy().astype(np.float32)
    round_trip = float(np.abs(wav[:min(len(wav), n)] - signal[:min(len(wav), n)]).max())

    # Resample 48 k -> 44.1 k for the same signal, for the dsp.Resample port.
    resampled = torchaudio_resample(signal, 48000, 44100)

    start = len(blob)
    blob += signal.astype("<f4").tobytes()
    blob += np.ascontiguousarray(magnitude[0].numpy()).astype("<f4").tobytes()
    blob += np.ascontiguousarray(phase[0].numpy()).astype("<f4").tobytes()
    blob += wav.astype("<f4").tobytes()
    blob += resampled.astype("<f4").tobytes()
    index.append({"n": n, "frames": int(frames), "offset": start,
                  "bytes": len(blob) - start, "wav_len": int(wav.shape[0]),
                  "resampled_len": int(resampled.shape[0]),
                  "round_trip_max_abs": round_trip})
    print(f"n={n} frames={frames} wav={wav.shape[0]} resampled={resampled.shape[0]} round_trip={round_trip:.3e}")

Path("testdata/dsp_stft.bin").write_bytes(bytes(blob))
Path("testdata/dsp_stft.json").write_text(
    json.dumps({"cases": index, "n_fft": N_FFT, "hop": HOP}, separators=(",", ":")) + "\n",
    encoding="utf-8")
print("blob", len(blob))


