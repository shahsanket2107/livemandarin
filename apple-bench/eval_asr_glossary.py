"""Glossary strategies for Qwen3-ASR: none / static list / per-utterance phonetic retrieval.

Metrics: CER vs reference, and term recall on the jargon set (fraction of glossary terms in
the reference that appear verbatim, case-insensitive, in the output).
Usage: ../server/.venv/bin/python eval_asr_glossary.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from lexicon import Lexicon  # noqa: E402
from score import edits, norm  # noqa: E402

MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
glossary = yaml.safe_load((Path(__file__).parents[1] / "server/glossary.yaml").read_text())
STATIC = " ".join(glossary["asr_hints"])
jargon_terms = sorted({t for line in open("jargon.jsonl") for t in json.loads(line)["terms"]})
# Full lexicon: every English hint, every Chinese glossary term, every term the meetings use.
LEXICON = Lexicon(glossary["asr_hints"] + [t["zh"] for t in glossary["terms"]] + jargon_terms)

rows = [json.loads(l) for l in open("data/manifest_jargon.jsonl")]
rows += [json.loads(l) for l in open("data/manifest.jsonl")]


def main() -> None:
    from mlx_qwen3_asr import Session

    session = Session(model=MODEL)
    session.transcribe(np.zeros(16000, dtype=np.float32))
    results = {k: {} for k in ("none", "static", "retrieval")}
    for n, row in enumerate(rows):
        audio, _ = sf.read(row["wav"], dtype="float32")
        t0 = time.perf_counter()
        first = session.transcribe(audio).text
        t_none = time.perf_counter() - t0
        t0 = time.perf_counter()
        static = session.transcribe(audio, context=STATIC).text
        t_static = time.perf_counter() - t0
        t0 = time.perf_counter()
        picked = LEXICON.select(first)
        retrieval = session.transcribe(audio, context=" ".join(picked)).text if picked else first
        t_retr = time.perf_counter() - t0 + t_none  # two passes when something matched
        for name, text, secs in (("none", first, t_none), ("static", static, t_static),
                                 ("retrieval", retrieval, t_retr)):
            results[name][row["id"]] = {"text": text, "s": secs, "picked": picked if name == "retrieval" else None}
        if n % 50 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    Path("data/asr_glossary_out.json").write_text(json.dumps(results, ensure_ascii=False))
    report(results)


def report(results) -> None:
    sets = sorted({r["set"] for r in rows})
    print(f"\n{'set':16}" + "".join(f"{k:>22}" for k in results))
    for s in sets:
        subset = [r for r in rows if r["set"] == s]
        cells = []
        for name, res in results.items():
            err = sum(edits(norm(r["ref_zh"]), norm(res[r["id"]]["text"])) for r in subset)
            chars = sum(len(norm(r["ref_zh"])) for r in subset)
            secs = sum(res[r["id"]]["s"] for r in subset) / len(subset)
            cell = f"{100 * err / chars:5.1f}% {secs:4.2f}s"
            if "terms" in subset[0]:
                hit = sum(t.lower() in res[r["id"]]["text"].lower() for r in subset for t in r["terms"])
                total = sum(len(r["terms"]) for r in subset)
                cell += f" T{100 * hit / total:3.0f}%"
            cells.append(cell)
        print(f"{s:16}" + "".join(f"{c:>22}" for c in cells))
    print("(CER, seconds/utterance, T = term recall on the jargon set)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.loads(Path("data/asr_glossary_out.json").read_text()))
    else:
        main()
