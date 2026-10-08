"""Dump small CPU parity values from the read-only Irodori reference.

Run with the reference venv's python, passing the Irodori-TTS checkout as the
first argument. No checkpoint or GPU is loaded.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from irodori_tts.model import RMSNorm, LowRankAdaLN, apply_rotary_emb, get_timestep_embedding, precompute_freqs_cis
from irodori_tts.rf import temporal_score_rescale
from irodori_tts.duration import build_duration_features

torch.set_num_threads(1)
x = torch.tensor([0.125, -0.5, 1.25, -2.0, 0.75, 0.25, -0.125, 2.5], dtype=torch.float32)
norm = RMSNorm(4, eps=1e-6)
with torch.no_grad():
    norm.weight.copy_(torch.tensor([0.75, 1.25, -0.5, 2.0]))
rotary_input = torch.arange(48, dtype=torch.float32).reshape(1, 3, 2, 8) / 17 - 1
times = torch.tensor([0.999, 0.5, 0.0], dtype=torch.float32)
velocity = torch.tensor([0.2, -0.4, 0.6, 0.8], dtype=torch.float32)
latent = torch.tensor([-0.1, 0.3, 0.2, -0.5], dtype=torch.float32)

def schedule(steps, mode, sway):
    u = torch.linspace(0, 1, steps + 1)
    if mode == "sway":
        u = (u + sway * (torch.cos(0.5 * torch.pi * u) + u - 1)).clamp(0, 1)
    return ((1 - u) * 0.999).tolist()

fixture = {
    "rms_input": x.tolist(),
    "rms_weight": norm.weight.detach().tolist(),
    "rms_output": norm(x.reshape(2, 4)).flatten().tolist(),
    "rotary_input": rotary_input.flatten().tolist(),
    "rotary_output": apply_rotary_emb(rotary_input, precompute_freqs_cis(8, 3)).flatten().tolist(),
    "timestep_input": times.tolist(),
    "timestep_output": get_timestep_embedding(times, 8).flatten().tolist(),
    "schedule_linear": schedule(8, "linear", -1),
    "schedule_sway": schedule(8, "sway", -1),
    "rescale_velocity": velocity.tolist(),
    "rescale_latent": latent.tolist(),
    "rescale_output": temporal_score_rescale(velocity, latent, 0.6, 1.2, 0.7).tolist(),
    "duration_texts": ["こんにちは。漢字ABC😮‍💨", "おーい、テスト…!", ""],
    "duration_tokens": [14, 9, 1],
    "duration_speakers": [True, False, True],
}
fixture["duration_outputs"] = build_duration_features(
    fixture["duration_texts"], token_counts=fixture["duration_tokens"],
    max_text_len=256, has_speaker=fixture["duration_speakers"]
).tolist()
torch.manual_seed(714)
adaln = LowRankAdaLN(8, 3, 1e-5).eval()
with torch.no_grad():
    for name, param in adaln.named_parameters():
        param.copy_(torch.randn_like(param) * 0.1)
adaln_x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 7 - 1
adaln_cond = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 19 - 1
adaln_y, adaln_gate = adaln(adaln_x, adaln_cond)
fixture["adaln_input"] = adaln_x.flatten().tolist()
fixture["adaln_cond"] = adaln_cond.flatten().tolist()
fixture["adaln_output"] = adaln_y.flatten().tolist()
fixture["adaln_gate"] = adaln_gate.flatten().tolist()
fixture["adaln_state"] = {name: param.detach().flatten().tolist() for name, param in adaln.named_parameters()}
Path("testdata/irodori_primitives.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
