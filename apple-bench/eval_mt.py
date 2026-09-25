"""Translation bake-off (zh→en): quality, glossary-term accuracy, latency, memory.

Candidates run in-process with MLX (one model loaded at a time, greedy decoding):
  Hy-MT2-1.8B 4-bit / 8-bit, MiLMMT-46-4B 5-bit, TranslateGemma-4B 4-bit.
Glossary strategies:
  plain   – source as-is
  presub  – glossary Chinese terms replaced by their English form in the source before translating
  terms   – Hy-MT2's official terminology template (Hy-MT2 only)
Data: 150 FLEURS sentences (human English refs) + 40 jargon sentences (my refs + expected terms).
Writes data/mt_<name>.jsonl; `--report` prints the table (also folds in Apple results if present).
Usage: .venv/bin/python eval_mt.py [--report]
"""

import json
import re
import sys
import time
from pathlib import Path

import sacrebleu
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from mt import HUNYUAN, HY_DENSE, GlossaryTerm, format_chat  # noqa: E402

ROOT = Path(__file__).resolve().parent
glossary = yaml.safe_load((ROOT.parent / "server/glossary.yaml").read_text())
TERMS = [GlossaryTerm(str(t["zh"]), str(t["en"])) for t in glossary["terms"]]
ZH2EN = {t.zh: t.en for t in TERMS}


def presub(text: str) -> str:
    """Replace glossary Chinese terms with English, longest first (最大回撤 before 回撤)."""
    for t in sorted(TERMS, key=lambda t: -len(t.zh)):
        text = text.replace(t.zh, f" {t.en} ")
    return re.sub(r"\s+", " ", text).strip()


def expected_terms(terms: list[str]) -> list[str]:
    return [ZH2EN.get(t, t) for t in terms]


def dataset() -> list[dict]:
    rows = []
    for line in open(ROOT / "data/manifest.jsonl"):
        r = json.loads(line)
        if r["set"] == "fleurs":
            rows.append({"id": r["id"], "set": "fleurs", "zh": r["ref_zh"], "en": r["ref_en"], "terms": []})
    for i, line in enumerate(open(ROOT / "jargon.jsonl")):
        r = json.loads(line)
        rows.append({"id": f"j{i:02d}", "set": "jargon", "zh": r["zh"], "en": r["en"],
                     "terms": expected_terms(r["terms"])})
    return rows


CANDIDATES = {
    "hymt2-1.8b-4bit": "mlx-community/Hy-MT2-1.8B-4bit",
    "hymt2-1.8b-8bit": "mlx-community/Hy-MT2-1.8B-8bit",
    "milmmt-4b-5bit": "translate-studio/MiLMMT-46-4B-v1.0-5bit-MLX",
    "translategemma-4b-4bit": "mlx-community/translategemma-4b-it-4bit",
    "hymt2-7b-4bit": "mlx-community/Hy-MT2-7B-4bit",
    "seedx-ppo-7b-4bit": "dong-99/Seed-X-PPO-7B-mlx-4Bit",
    "milmmt-12b-4bit": "mlx-community/MiLMMT-46-12B-v0.1-4bit",
}


def make_prompt(name: str, tokenizer, zh: str, use_terms: bool) -> str:
    if name.startswith("hymt2"):
        fmt = HUNYUAN if "7b" in name else HY_DENSE
        return format_chat(fmt, zh, [], TERMS if use_terms else [])
    if name.startswith("seedx"):
        return f"Translate the following Chinese sentence into English:\n{zh} <en>"
    if name.startswith("milmmt"):
        return f"Translate this from Chinese (Simplified) to English:\nChinese (Simplified): {zh}\nEnglish:"
    messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": "zh",
                                              "target_lang_code": "en", "text": zh}]}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def run(name: str, repo: str) -> None:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    mx.reset_peak_memory()
    model, tokenizer = load(repo)
    sampler = make_sampler(temp=0.0)
    strategies = ["plain", "presub"] + (["terms"] if name.startswith("hymt2") else [])
    big = "7b" in name or "12b" in name  # large models: news as-is, jargon with pre-substitution only
    generate(model, tokenizer, prompt=make_prompt(name, tokenizer, "你好。", False), max_tokens=8, sampler=sampler)
    with open(ROOT / f"data/mt_{name}.jsonl", "w") as out:
        for row in dataset():
            for strat in strategies:
                if big and strat != ("plain" if row["set"] == "fleurs" else "presub"):
                    continue
                src = presub(row["zh"]) if strat == "presub" else row["zh"]
                prompt = make_prompt(name, tokenizer, src, strat == "terms")
                t0 = time.perf_counter()
                text = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler)
                secs = time.perf_counter() - t0
                for marker in ("<end_of_turn>", "<|eos|>", "</s>"):
                    text = text.split(marker)[0]
                text = text.split("\n")[0].strip() if name.startswith(("milmmt", "seedx")) else text.strip()
                out.write(json.dumps({"id": row["id"], "strategy": strat, "en": text, "s": secs},
                                     ensure_ascii=False) + "\n")
    peak = mx.get_peak_memory() / 1e9
    (ROOT / f"data/mt_{name}.mem").write_text(f"{peak:.2f}")
    print(f"{name}: done, peak {peak:.2f} GB", flush=True)


def report() -> None:
    rows = {r["id"]: r for r in dataset()}
    results: dict[tuple[str, str], dict] = {}
    for path in sorted((ROOT / "data").glob("mt_*.jsonl")):
        name = path.stem[3:]
        if name == "rows":
            continue
        for line in open(path):
            r = json.loads(line)
            results.setdefault((name, r["strategy"]), {})[r["id"]] = r
    print(f"{'translator':26}{'glossary':>9}{'FLEURS chrF':>13}{'jargon chrF':>13}{'term acc':>10}{'s/sent':>8}{'peak GB':>9}")
    for (name, strat), res in sorted(results.items()):
        mem = ROOT / f"data/mt_{name}.mem"
        cells = []
        for s in ("fleurs", "jargon"):
            ids = [i for i, r in rows.items() if r["set"] == s and i in res]
            if not ids:
                cells.append(float("nan"))
                continue
            cells.append(sacrebleu.corpus_chrf([res[i]["en"] for i in ids], [[rows[i]["en"] for i in ids]]).score)
        jids = [i for i, r in rows.items() if r["set"] == "jargon" and i in res]
        hit = sum(t.lower() in res[i]["en"].lower() for i in jids for t in rows[i]["terms"])
        total = sum(len(rows[i]["terms"]) for i in jids)
        secs = sum(r["s"] for r in res.values()) / len(res)
        print(f"{name:26}{strat:>9}{cells[0]:>13.1f}{cells[1]:>13.1f}{100 * hit / max(total, 1):>9.0f}%{secs:>8.2f}"
              f"{(mem.read_text() if mem.exists() else '-'):>9}")


def run_ollama() -> None:
    """Current production translator (Ollama, Q8 GGUF), same prompts, for a like-for-like baseline."""
    import asyncio

    from mt import Translator

    async def go():
        t = Translator("http://127.0.0.1:11434", "hf.co/tencent/Hy-MT2-1.8B-GGUF:Q8_0")
        await t.load()
        with open(ROOT / "data/mt_hymt2-1.8b-ollama-q8.jsonl", "w") as out:
            for row in dataset():
                for strat in ("plain", "presub", "terms"):
                    src = presub(row["zh"]) if strat == "presub" else row["zh"]
                    t0 = time.perf_counter()
                    text = await t.translate(src, [], TERMS if strat == "terms" else [])
                    out.write(json.dumps({"id": row["id"], "strategy": strat, "en": text,
                                          "s": time.perf_counter() - t0}, ensure_ascii=False) + "\n")
        (ROOT / "data/mt_hymt2-1.8b-ollama-q8.mem").write_text("2.13")  # Ollama-reported size

    asyncio.run(go())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ollama":
        run_ollama()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "--dump":  # rows for the Apple runner
        for r in dataset():
            for strat in ("plain", "presub"):
                src = presub(r["zh"]) if strat == "presub" else r["zh"]
                protect = [ZH2EN[t.zh] for t in TERMS if t.zh in r["zh"]] if strat == "presub" else []
                print(json.dumps({"id": r["id"], "strategy": strat, "src": src, "protect": protect}, ensure_ascii=False))
    else:
        for name, repo in CANDIDATES.items():
            if len(sys.argv) > 1 and name not in sys.argv[1:]:
                continue
            run(name, repo)
