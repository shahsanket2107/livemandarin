"""Phonetic glossary retrieval for the recognizer.

A long static hint list measurably hurts Qwen3-ASR, so instead: transcribe once, find the
glossary entries that *sound like* something in the transcript (toneless pinyin for Chinese,
letters for English), and re-transcribe with only those few entries as context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pypinyin import lazy_pinyin

LATIN = re.compile(r"[A-Za-z]+")


def phonetic(text: str) -> str:
    """Toneless pinyin for Han characters, lowercase letters for Latin, digits kept."""
    out = []
    for token in lazy_pinyin(text, errors=lambda chars: [c for c in chars]):
        out.append(re.sub(r"[^a-z0-9]", "", token.lower()))
    return "".join(out)


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_window(key: str, text_key: str) -> float:
    """Highest similarity (0–1) between `key` and any substring of `text_key` of similar length."""
    if not key or not text_key:
        return 0.0
    n, best = len(key), 0.0
    for size in {max(1, n - 2), n - 1, n, n + 1, n + 2}:
        if size <= 0:
            continue
        for start in range(0, max(1, len(text_key) - size + 1)):
            window = text_key[start:start + size]
            best = max(best, 1 - _distance(key, window) / max(n, len(window)))
            if best == 1.0:
                return best
    return best


@dataclass(frozen=True)
class Entry:
    surface: str  # what should appear in the transcript
    key: str      # phonetic key


class Lexicon:
    def __init__(self, entries: list[str]):
        seen, self.entries = set(), []
        for surface in entries:
            if surface and surface not in seen:
                seen.add(surface)
                self.entries.append(Entry(surface, phonetic(surface)))

    def select(self, transcript: str, limit: int = 8, threshold: float = 0.6) -> list[str]:
        """Glossary entries that sound like part of `transcript`, best matches first."""
        text_key = phonetic(transcript)
        scored = []
        for entry in self.entries:
            if len(entry.key) < 2:
                continue
            score = best_window(entry.key, text_key)
            if score >= threshold:
                scored.append((score, len(entry.key), entry.surface))
        scored.sort(reverse=True)
        return [surface for _, _, surface in scored[:limit]]
