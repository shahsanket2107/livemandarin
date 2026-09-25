"""Run the current stack (Qwen3-ASR + Hy-MT2) over the same manifest as applerun.

Output line: id, set, qwen_zh, asr_s, and Hy-MT2 translations of the reference Chinese,
Apple's Chinese and Qwen's Chinese (FLEURS only; ASCEND has no English references).
Usage: ../server/.venv/bin/python ours.py data/manifest.jsonl data/apple_out.jsonl data/ours_out.jsonl
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from mt import Translator  # noqa: E402

MT_MODEL = "hf.co/tencent/Hy-MT2-1.8B-GGUF:Q8_0"


async def main() -> None:
    from mlx_qwen3_asr import Session

    manifest = [json.loads(line) for line in open(sys.argv[1])]
    apple = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
    session = Session(model="mlx-community/Qwen3-ASR-1.7B-8bit")
    session.transcribe(np.zeros(16000, dtype=np.float32))
    translator = Translator("http://127.0.0.1:11434", MT_MODEL)
    await translator.load()

    with open(sys.argv[3], "w") as out:
        for n, item in enumerate(manifest):
            audio, _ = sf.read(item["wav"], dtype="float32")
            t0 = time.perf_counter()
            result = session.transcribe(audio)  # no glossary: datasets are unrelated to it
            row = {"id": item["id"], "set": item["set"], "qwen_zh": result.text.strip(),
                   "qwen_lang": result.language, "asr_s": time.perf_counter() - t0}
            if item["set"] == "fleurs":
                t1 = time.perf_counter()
                row["hymt_en_from_ref"] = await translator.translate(item["ref_zh"], [], [])
                row["mt_s"] = time.perf_counter() - t1
                row["hymt_en_from_qwen"] = await translator.translate(row["qwen_zh"], [], []) if row["qwen_zh"] else ""
                az = apple.get(item["id"], {}).get("apple_zh", "")
                row["hymt_en_from_apple"] = await translator.translate(az, [], []) if az else ""
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 25 == 0:
                print(f"{n}/{len(manifest)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
