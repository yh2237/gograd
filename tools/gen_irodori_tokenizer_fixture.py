"""Generate token-ID-only parity data using the cached ModernBERT-ja tokenizer."""

import glob
import importlib.util
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

root = Path(os.environ.get("IRODORI_REFERENCE", r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS"))
cache = Path(os.environ.get("HF_HOME", r"D:\project\UtauTTS\out\reference-tts-20261008b\hf-cache"))
tokenizer_dir = Path(glob.glob(str(cache / "hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/*/tokenizer"))[0])
spec = importlib.util.spec_from_file_location("irodori_tokenizer", root / "irodori_tts/tokenizer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
wrapper = module.PretrainedTextTokenizer(AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True))
normalization = json.loads(Path("testdata/irodori_text.json").read_text(encoding="utf-8"))
texts = [item["output"] for item in normalization]
starts = ["今日は", "明日は", "おはよう", "さようなら", "これは", "静かな夜に", "歌声が", "東京駅で", "ひらがなとカタカナ", "123ABC", "外来語のテスト", "もう一度"]
ends = ["晴れです。", "雨が降ります！", "楽しいですね？", "…少し待って。", "元気です。", "ありがとう♡"]
texts += [starts[i % len(starts)] + ends[(i * 5) % len(ends)] + f" 第{i + 1}回" + (" 確認" if i % 3 == 0 else "") for i in range(50)]
texts += ["あ" * 300, "test " * 100, "漢字かな交じり文。" * 50]
cases = []
for s in texts:
    record = {"text": s, "ids": wrapper.encode(s).tolist()}
    for length in (1, 8, 32, 256):
        ids, _ = wrapper.batch_encode([s], max_length=length)
        record[f"padded_{length}"] = ids[0].tolist()
    cases.append(record)
Path("testdata/irodori_tokenizer.json").write_text(json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
print(f"wrote {len(cases)} token-ID fixtures from {tokenizer_dir}")
