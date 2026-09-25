"""Score Apple vs current stack. Usage: ../server/.venv/bin/python score.py

Recognition: character error rate (CER) vs reference transcripts, after normalizing
punctuation, spacing, case and full/half-width forms; English words count per character.
Translation: chrF (0–100, higher is better) vs FLEURS human English references.
"""

import json
import re
import unicodedata
from collections import defaultdict

import sacrebleu


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[\W_]+", "", text)


def edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


man = {r["id"]: r for r in map(json.loads, open("data/manifest.jsonl"))}
apple, ours = load("data/apple_out.jsonl"), load("data/ours_out.jsonl")
apple_q = load("data/apple_qwen_out.jsonl") if __import__("os").path.exists("data/apple_qwen_out.jsonl") else {}

print("RECOGNITION — character error rate (lower is better)")
stats = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0])  # apple_err, qwen_err, chars, apple_s, qwen_s, n
for i, m in man.items():
    if i not in apple or i not in ours:
        continue
    ref = norm(m["ref_zh"])
    s = stats[m["set"]]
    s[0] += edits(ref, norm(apple[i].get("apple_zh", "")))
    s[1] += edits(ref, norm(ours[i]["qwen_zh"]))
    s[2] += len(ref)
    s[3] += apple[i].get("asr_s", 0)
    s[4] += ours[i]["asr_s"]
    s[5] += 1
print(f"{'set':16}{'n':>4}{'Apple':>9}{'Qwen':>9}{'Apple s/utt':>13}{'Qwen s/utt':>12}")
for name, (ae, qe, c, at, qt, n) in sorted(stats.items()):
    print(f"{name:16}{n:>4}{100*ae/c:>8.1f}%{100*qe/c:>8.1f}%{at/n:>13.2f}{qt/n:>12.2f}")

print("\nTRANSLATION — chrF vs human English (FLEURS, higher is better)")
ids = [i for i, m in man.items() if m["set"] == "fleurs" and i in ours and i in apple]
refs = [[man[i]["ref_en"] for i in ids]]
combos = {
    "perfect Chinese → Apple": [apple[i].get("apple_en_from_ref", "") for i in ids],
    "perfect Chinese → Hy-MT2": [ours[i]["hymt_en_from_ref"] for i in ids],
    "Qwen → Hy-MT2 (current)": [ours[i]["hymt_en_from_qwen"] for i in ids],
    "Apple → Hy-MT2": [ours[i]["hymt_en_from_apple"] for i in ids],
    "Apple → Apple": [apple[i].get("apple_en_from_asr", "") for i in ids],
}
if apple_q:
    combos["Qwen → Apple"] = [apple_q.get(i, {}).get("apple_en", "") for i in ids]
for name, hyp in combos.items():
    print(f"  {name:26} chrF {sacrebleu.corpus_chrf(hyp, refs).score:5.1f}   BLEU {sacrebleu.corpus_bleu(hyp, refs).score:5.1f}")
mt_a = sum(apple[i].get("mt_s", 0) for i in ids) / len(ids)
mt_h = sum(ours[i]["mt_s"] for i in ids) / len(ids)
print(f"\n  translation time per sentence: Apple {mt_a:.2f}s, Hy-MT2 {mt_h:.2f}s")
