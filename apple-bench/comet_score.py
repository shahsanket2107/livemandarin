"""COMET-22 (reference-based, Unbabel/wmt22-comet-da) for every translator in data/mt_*.jsonl.

Scores each (translator, strategy) on FLEURS and on the jargon set. COMET judges meaning, so it
does not penalize correct paraphrases the way chrF does. Results cached in data/comet.json.
Usage: .venv-comet/bin/python comet_score.py
"""

import json
from pathlib import Path

from comet import download_model, load_from_checkpoint

ROOT = Path(__file__).resolve().parent
man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/manifest.jsonl")}
src = {i: {"zh": m["ref_zh"], "en": m["ref_en"], "set": "fleurs"} for i, m in man.items() if m["set"] == "fleurs"}
for k, line in enumerate(open(ROOT / "jargon.jsonl")):
    r = json.loads(line)
    src[f"j{k:02d}"] = {"zh": r["zh"], "en": r["en"], "set": "jargon"}

cache_path = ROOT / "data/comet.json"
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

for path in sorted((ROOT / "data").glob("mt_*.jsonl")):
    name = path.stem[3:]
    if name == "rows":
        continue
    groups: dict[str, dict] = {}
    for r in map(json.loads, open(path)):
        groups.setdefault(r["strategy"], {})[r["id"]] = r["en"]
    for strat, hyps in groups.items():
        for s in ("fleurs", "jargon"):
            key = f"{name}|{strat}|{s}"
            ids = [i for i in hyps if src.get(i, {}).get("set") == s]
            if not ids or (key in cache and cache[key]["n"] == len(ids)):
                continue
            data = [{"src": src[i]["zh"], "mt": hyps[i], "ref": src[i]["en"]} for i in ids]
            out = model.predict(data, batch_size=16, gpus=0, progress_bar=False, num_workers=1)
            cache[key] = {"n": len(ids), "comet": round(100 * out.system_score, 1)}
            cache_path.write_text(json.dumps(cache, indent=1))
            print(key, cache[key], flush=True)

print(f"\n{'translator':26}{'glossary':>9}{'FLEURS COMET':>14}{'jargon COMET':>14}")
names = sorted({k.rsplit("|", 1)[0] for k in cache})
for nk in names:
    name, strat = nk.split("|")
    f = cache.get(f"{nk}|fleurs", {}).get("comet", float("nan"))
    j = cache.get(f"{nk}|jargon", {}).get("comet", float("nan"))
    print(f"{name:26}{strat:>9}{f:>14.1f}{j:>14.1f}")
