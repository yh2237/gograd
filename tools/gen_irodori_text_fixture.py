"""Generate small normalization parity cases from the read-only Irodori source."""

import importlib.util
import json
import os
from pathlib import Path

reference = Path(os.environ.get(
    "IRODORI_REFERENCE",
    r"D:\project\UtauTTS\out\reference-tts-20261008b\Irodori-TTS",
))
spec = importlib.util.spec_from_file_location(
    "irodori_text_normalization", reference / "irodori_tts" / "text_normalization.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = [
    "「（ﾃｽﾄ！）...」", "【「今日は〜いい天気です？」】", "　[ n ]\t[n]\u3000", 
    "『あ…...…い』", "「あ」い", "(abc)", "(a)(b)", "㌔㍍①②－―", 
    "\n  あ  \n", "(（あ）)", "《VOICE▼ACTRESS》", "⏩[n]こんにちは！",
]
output = [{"input": s, "output": module.normalize_text(s)} for s in cases]
Path("testdata/irodori_text.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
