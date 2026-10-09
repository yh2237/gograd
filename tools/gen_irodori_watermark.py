"""SilentCipher watermark reference for gograd/irodori.

Runs the reference's own encode_wav path on the same inputs Go reads, so the port
can be checked without Python at test time. Requires the converted checkpoint in
IRODORI_WATERMARK_SAFE and the venv's silentcipher package.

The encoded carrier is recorded for a spread of channels rather than all 32, and
the concatenated decoder input is not recorded at all because it is exactly the
coded carrier repeated alongside the carrier itself. That keeps the fixture small
enough to commit while still exercising every convolution and normalization.
"""

import json
from pathlib import Path

import numpy as np
import torch

sys_path = r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"

CASES = [
    # (samples, sample rate, payload seed)
    (48000 * 3, 48000, 5),
    (48000 * 3 + 37, 48000, 11),
    (44100, 44100, 13),
    (48000, 48000, 7),
]
PAYLOAD = [73, 82, 68, 84, 83]
DIM = 5
SAMPLED_CHANNELS = [0, 8, 16, 24, 31]


def stft_transform(stft_singleton, x, n_fft):
    x = torch.nn.functional.pad(x, (0, n_fft - x.shape[1] % n_fft))
    fft = torch.stft(x, n_fft, x.shape[1] and 2048, n_fft,
                     window=torch.hann_window(n_fft), return_complex=True)
    real, imag = fft.real, fft.imag
    squared = real**2 + imag**2
    eps = torch.ones_like(squared) * (squared == 0).float() * 1e-24
    magnitude = torch.sqrt(squared + eps) - torch.sqrt(eps)
    return magnitude, torch.atan2(imag, real)


def run(model, audio, sample_rate, payload):
    import torchaudio

    vector = torch.from_numpy(audio.copy()).unsqueeze(0)
    original = vector.clone()
    y = vector
    if sample_rate != model.sr:
        y = torchaudio.functional.resample(y.view(1, -1), orig_freq=sample_rate,
                                          new_freq=model.sr).squeeze()
    else:
        y = y.squeeze()
    power = torch.mean(y**2)
    y = y * torch.sqrt(torch.tensor(model.average_energy_VCTK) / power)
    y = y.unsqueeze(0).unsqueeze(0)
    carrier, phase = model.stft.transform(y.squeeze(1))
    carrier = carrier[:, None]
    phase = phase[:, None]

    def binary_encode(mes):
        bits = ''.join(['{0:08b}'.format(v) for v in mes])
        return [int(bits[i*2:i*2+2], 2) for i in range(len(bits)//2)]

    symbols = binary_encode(payload)
    index = np.concatenate((np.array(symbols) + 1, [0]))
    one_hot = np.identity(model.message_dim)[index]
    frames = carrier.shape[3]
    full, extra = frames // model.message_len, frames % model.message_len
    grid = np.tile(one_hot.T, (1, full))
    if extra:
        grid = np.concatenate([grid, one_hot.T[:, :extra]], axis=1)
    # letters_encoding stacks one entry per message before encode_wav adds its own.
    msg = torch.tensor(np.stack([grid]), dtype=torch.float32).unsqueeze(0)

    carrier_enc = model.enc_c(carrier)
    msg_enc = model.enc_c.transform_message(msg)
    merged = torch.cat((carrier_enc, carrier.repeat(1, model.encoder_out_dim, 1, 1),
                        msg_enc.repeat(1, model.encoder_out_dim, 1, 1)), dim=1)
    with torch.no_grad():
        raw = model.dec_c(merged, model.config.message_sdr)
    info = raw * (torch.mean(carrier**2, dim=(2, 3), keepdim=True)**0.5)
    info = -info
    rebuilt = torch.nn.functional.relu(info + carrier)

    # Verify the Go port against the reference's own entry point as well.
    reference, _ = model.encode_wav(torch.from_numpy(original[0].numpy().copy()),
                                    sample_rate, payload, calc_sdr=False)
    reference = reference.detach().numpy().reshape(-1).astype(np.float32)

    sampled = carrier_enc.detach().numpy()[0][SAMPLED_CHANNELS]
    stages = {
        "audio": audio,
        "out": reference,
        "carrier": carrier.detach().numpy()[0, 0],
        "rebuilt": rebuilt.detach().numpy()[0, 0],
        "sampled": np.ascontiguousarray(sampled),
        "raw": raw.detach().numpy()[0, 0],
        "channels": SAMPLED_CHANNELS,
    }
    return stages, frames


def main():
    import silentcipher

    model = silentcipher.get_model(model_type="44.1k", device="cpu")
    n_fft, message_len = model.config.N_FFT, model.message_len
    if message_len != 21 or len(PAYLOAD) * 8 // 2 + 1 != message_len:
        raise SystemExit(f"payload does not fit message_len={message_len}")
    blob = bytearray()
    index = []
    for n, rate, seed in CASES:
        rng = np.random.default_rng(seed)
        audio = (np.sin(np.arange(n) * 0.021) * 0.35 + rng.standard_normal(n) * 0.02).astype(np.float32)
        stages, frames = run(model, audio, rate, PAYLOAD)
        bins = int(stages["carrier"].shape[0])
        delta = float(np.abs(stages["out"][:n] - audio[:n]).max())
        start = len(blob)
        for name in ("audio", "out", "carrier", "rebuilt", "sampled", "raw"):
            blob += stages[name].astype("<f4").tobytes()
        layout = [{"field": name, "elements": int(np.prod(stages[name].shape))}
                  for name in ("audio", "out", "carrier", "rebuilt", "sampled", "raw")]
        index.append({
            "n": n, "rate": rate, "seed": seed, "offset": start,
            "bytes": len(blob) - start, "out_len": int(stages["out"].shape[0]),
            "bins": bins, "frames": frames, "layout": layout,
            "channels": stages["channels"], "delta_max_abs": delta,
        })
        print(f"n={n} rate={rate} frames={frames} delta={delta:.3e}")
    Path("testdata/irodori_watermark.bin").write_bytes(bytes(blob))
    Path("testdata/irodori_watermark.json").write_text(
        json.dumps({"cases": index}, separators=(",", ":")) + "\n", encoding="utf-8")
    print("blob", len(blob))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, sys_path)
    main()
