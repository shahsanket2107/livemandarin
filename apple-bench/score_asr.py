"""Compare recognizers: Qwen3-ASR (none/static hints) vs candidates in data/asrm_*.jsonl.
Usage: ../server/.venv/bin/python score_asr.py"""
import json
from pathlib import Path

from score import edits, norm

rows = [json.loads(l) for l in open("data/manifest_jargon.jsonl")] + [json.loads(l) for l in open("data/manifest.jsonl")]
qwen = json.loads(Path("data/asr_glossary_out.json").read_text())
systems = {"Qwen3-ASR (no hints)": qwen["none"], "Qwen3-ASR (static hints)": qwen["static"]}
for p in sorted(Path("data").glob("asrm_*.jsonl")):
    res = {r["id"]: r for r in map(json.loads, open(p))}
    if len(res) == len(rows):
        mem = p.with_suffix(".mem")
        systems[p.stem[5:] + (f" [{mem.read_text()} GB]" if mem.exists() else "")] = res
sets = sorted({r["set"] for r in rows})
print(f"{'recognizer':34}" + "".join(f"{s:>16}" for s in sets) + f"{'s/utt':>8}")
for name, res in systems.items():
    cells = []
    for s in sets:
        sub = [r for r in rows if r["set"] == s]
        err = sum(edits(norm(r["ref_zh"]), norm(res[r["id"]]["text"])) for r in sub)
        cell = f"{100 * err / sum(len(norm(r['ref_zh'])) for r in sub):.1f}%"
        if "terms" in sub[0]:
            hit = sum(t.lower() in res[r["id"]]["text"].lower() for r in sub for t in r["terms"])
            cell += f" T{100 * hit / sum(len(r['terms']) for r in sub):.0f}"
        cells.append(cell)
    secs = sum(res[r["id"]]["s"] for r in rows) / len(rows)
    print(f"{name:34}" + "".join(f"{c:>16}" for c in cells) + f"{secs:>8.2f}")
print("CER (lower better); T = glossary-term recall %; timings are contended (tests ran concurrently)")
