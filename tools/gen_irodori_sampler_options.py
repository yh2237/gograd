"""Irodori sampler option reference for gograd/irodori.

Runs the reference sampler with the optional truncation factor, temporal score
rescaling and speaker key/value scaling, so the Go port can be checked against
the reference's own loop rather than against a hand-written expectation.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS")

# Each case pairs a SamplerConfig-shaped override with the resulting final
# latent. Only the options the reference exposes are exercised.
CASES = [
    ("baseline", {}),
    ("truncation", {"truncation_factor": 0.85}),
    ("rescale", {"rescale_k": 1.2, "rescale_sigma": 0.7}),
    ("speaker_kv", {"speaker_kv_scale": 1.5}),
    ("speaker_kv_floor", {"speaker_kv_scale": 1.5, "speaker_kv_min_t": 0.8}),
    ("speaker_kv_layers", {"speaker_kv_scale": 1.25, "speaker_kv_max_layers": 4}),
    ("all", {"truncation_factor": 0.9, "rescale_k": 1.2, "rescale_sigma": 0.7,
             "speaker_kv_scale": 1.5, "speaker_kv_min_t": 0.8}),
]


def main():
    cache = Path(r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache")
    snapshot = Path(sorted(cache.glob("hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*"))[0])
    checkpoint = snapshot / "model.safetensors"

    import dataclasses
    from irodori_tts.config import ModelConfig
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.rf import sample_euler_rf_cfg

    with checkpoint.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(n))["__metadata__"]
    raw_cfg = json.loads(metadata["config_json"])
    cfg = ModelConfig(**{k: v for k, v in raw_cfg.items()
                         if k in {f.name for f in dataclasses.fields(ModelConfig)}})
    text_cfg = json.loads(metadata["text_encoder_config_json"])
    torch.set_num_threads(6)
    model = TextToLatentRFDiT(cfg, pretrained_backbone_config=text_cfg,
                              load_pretrained_backbone_weights=False).eval()
    from safetensors import safe_open
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader, torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(reader.get_tensor(name))

    from irodori_tts.tokenizer import PretrainedTextTokenizer
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(snapshot / "tokenizer"), add_bos=cfg.text_add_bos, local_files_only=True)
    text = "こんにちは、今日もいい天気ですね。"
    caption = "やさしい声で話す女性"
    ids, mask = tokenizer.batch_encode([text], max_length=24)
    cap_ids, cap_mask = tokenizer.batch_encode([caption], max_length=24)

    reference = Path(__file__).resolve().parent.parent / "testdata" / "irodori_reference_latent.f32"
    latent = np.fromfile(reference, dtype="<f4").reshape(1, -1, 32)
    ref = torch.from_numpy(latent.copy())
    ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool)

    common = dict(
        text_input_ids=ids, text_mask=mask, ref_latent=ref, ref_mask=ref_mask,
        caption_input_ids=cap_ids, caption_mask=cap_mask,
        sequence_length=4, num_steps=3, cfg_scale_text=3.0, cfg_scale_speaker=5.0,
        cfg_scale_caption=3.0, cfg_guidance_mode="independent", cfg_min_t=0.5,
        cfg_max_t=1.0, seed=7231, t_schedule_mode="sway", sway_coeff=-1.0,
        use_context_kv_cache=False,
    )

    out = {}
    for name, override in CASES:
        with torch.inference_mode():
            sampled = sample_euler_rf_cfg(model, **common, **override)
        values = sampled.reshape(-1).numpy().astype(np.float32)
        out[name] = values
        print(f"{name}: {len(values)} values, first {values[0]:.9g}")

    blob = bytearray()
    index = []
    for name, values in out.items():
        index.append({"name": name, "values": len(values), "offset": len(blob)})
        blob += values.astype("<f4").tobytes()
    outdir = Path(__file__).resolve().parent.parent / "testdata"
    (outdir / "irodori_sampler_options.bin").write_bytes(bytes(blob))
    (outdir / "irodori_sampler_options.json").write_text(
        json.dumps({"cases": index}, separators=(",", ":")) + "\n", encoding="utf-8")
    print("blob", len(blob))


if __name__ == "__main__":
    main()
