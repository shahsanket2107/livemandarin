"""Recognizer bake-off on FLEURS + ASCEND + jargon: FireRedASR2-AED (MLX), Fun-ASR-Nano (MLX,
with and without hotwords). Qwen3-ASR numbers come from eval_asr_glossary.py.

Usage: .venv/bin/python eval_asr_models.py [fireredasr2|funasr-nano|funasr-nano-hot ...]
Writes data/asrm_<name>.jsonl and data/asrm_<name>.mem (peak MLX memory, GB).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "server"))
from lexicon import Lexicon  # noqa: E402

glossary = yaml.safe_load((ROOT.parent / "server/glossary.yaml").read_text())
jargon_terms = sorted({t for line in open(ROOT / "jargon.jsonl") for t in json.loads(line)["terms"]})
LEXICON = Lexicon(glossary["asr_hints"] + [t["zh"] for t in glossary["terms"]] + jargon_terms)

rows = [json.loads(l) for l in open(ROOT / "data/manifest_jargon.jsonl")]
rows += [json.loads(l) for l in open(ROOT / "data/manifest.jsonl")]

CANDIDATES = {
    "fireredasr2": "mlx-community/FireRedASR2-AED-mlx",
    "funasr-nano": "mlx-community/Fun-ASR-Nano-2512",
    "funasr-nano-hot": "mlx-community/Fun-ASR-Nano-2512",
}


def run(name: str, repo: str) -> None:
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model

    mx.reset_peak_memory()
    model = load_model(repo)
    warm = mx.zeros(16000)
    model.generate(warm)
    with open(ROOT / f"data/asrm_{name}.jsonl", "w") as out:
        for n, row in enumerate(rows):
            audio = mx.array(sf.read(row["wav"], dtype="float32")[0])
            t0 = time.perf_counter()
            if name == "funasr-nano-hot":
                first = model.generate(audio).text
                picked = LEXICON.select(first)
                text = model.generate(audio, hotwords=picked).text if picked else first
            else:
                text = model.generate(audio).text
            secs = time.perf_counter() - t0
            out.write(json.dumps({"id": row["id"], "text": text.strip(), "s": secs}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"{name} {n}/{len(rows)}", flush=True)
    (ROOT / f"data/asrm_{name}.mem").write_text(f"{mx.get_peak_memory() / 1e9:.2f}")


if __name__ == "__main__":
    for name in sys.argv[1:] or CANDIDATES:
        run(name, CANDIDATES[name])
