"""Benchmark Hy-MT2 models and prompt styles on the labeled meeting script.

Usage: .venv/bin/python bench/bench_mt.py [--models ...] [--prompts plain context]
Requires `ollama serve` running with the models pulled. "context" feeds the model's own
previous translations as prior chat turns, exactly as the server does.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mt import GlossaryTerm, Translator, Turn  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
GLOSSARY = [GlossaryTerm("小王", "Xiao Wang"), GlossaryTerm("回滚", "rollback"), GlossaryTerm("上线", "release")]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["hf.co/tencent/Hy-MT2-1.8B-GGUF:Q8_0", "hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M"])
    parser.add_argument("--prompts", nargs="+", default=["plain", "context"])
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", default=str(Path(__file__).parent / "results_mt.json"))
    args = parser.parse_args()

    script = [json.loads(line) for line in (SAMPLES / "script.jsonl").open()]
    results = []
    for model in args.models:
        translator = Translator("http://127.0.0.1:11434", model, temperature=args.temperature)
        await translator.load()
        for style in args.prompts:
            rows, times = [], []
            history: deque[Turn] = deque(maxlen=args.history)
            for item in script:
                terms = GLOSSARY if style == "context" else []
                t0 = time.perf_counter()
                text = await translator.translate(item["zh"], list(history), terms)
                seconds = time.perf_counter() - t0
                if style == "context":
                    history.append(Turn(item["zh"], text))
                rows.append({"zh": item["zh"], "ref": item["en"], "hyp": text, "s": round(seconds, 3)})
                times.append(seconds)
            times.sort()
            r = {"model": model, "format": translator.format.name, "prompt": style,
                 "p50_s": round(times[len(times) // 2], 3), "max_s": round(times[-1], 3), "rows": rows}
            results.append(r)
            print(f"{model:40} {style:8} p50={r['p50_s']}s max={r['max_s']}s", flush=True)
        await translator.aclose()
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
