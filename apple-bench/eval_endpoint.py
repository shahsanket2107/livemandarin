"""Sentence endpointing: fixed-silence rules vs Silero + Smart Turn v3 gating.

Stream: the 150 FLEURS test sentences (complete sentences) joined with random 0.3–1.0 s gaps,
so every join is a true sentence end. A policy should cut at every join, as soon as possible,
and never inside a sentence.
Metrics: cuts inside sentences (fragments), joins with no cut (merged sentences), and mean
delay from the true end of speech to the cut.
Usage: .venv/bin/python eval_endpoint.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "server"))
from segmenter import WINDOW, SileroVAD  # noqa: E402

SR = 16000
rng = np.random.default_rng(3)
clips = [sf.read(json.loads(l)["wav"], dtype="float32")[0] for l in open(ROOT / "data/manifest.jsonl")
         if json.loads(l)["set"] == "fleurs"]
SILERO = ROOT.parent / "server/.venv/lib/python3.13/site-packages/silero_vad/data/silero_vad.onnx"


def trim(clip: np.ndarray) -> np.ndarray:
    """Cut the recording's own leading/trailing silence so the clip ends where speech ends."""
    v = SileroVAD(SILERO)
    p = [v(clip[i:i + WINDOW]) for i in range(0, len(clip) - WINDOW, WINDOW)]
    speech = [i for i, x in enumerate(p) if x >= 0.5]
    return clip[max(0, speech[0] - 2) * WINDOW:(speech[-1] + 2) * WINDOW] if speech else clip


clips = [trim(c) for c in clips]
parts, ends, pos = [], [], 0
for c in clips:
    parts.append(c)
    pos += len(c)
    ends.append(pos)
    gap = np.zeros(int(rng.uniform(0.3, 1.0) * SR), dtype=np.float32)
    parts.append(gap)
    pos += len(gap)
stream = np.concatenate(parts)

vad = SileroVAD(SILERO)
probs = np.array([vad(stream[i:i + WINDOW]) for i in range(0, len(stream) - WINDOW, WINDOW)])
print(f"stream {len(stream) / SR / 60:.1f} min, {len(ends)} sentences")

turn_path = next(Path.home().glob(".cache/huggingface/hub/models--pipecat-ai--smart-turn-v3/snapshots/*/smart-turn-v3.2-cpu.onnx"))
turn = ort.InferenceSession(str(turn_path), providers=["CPUExecutionProvider"])
extractor = WhisperFeatureExtractor(chunk_length=8)
turn_ms = []


def complete(audio: np.ndarray, threshold: float = 0.5) -> bool:
    audio = audio[-8 * SR:]
    feats = extractor(audio, sampling_rate=SR, return_tensors="np", padding="max_length",
                      max_length=8 * SR, truncation=True, do_normalize=True).input_features
    t0 = time.perf_counter()
    p = turn.run(None, {"input_features": feats.astype(np.float32)})[0].item()
    turn_ms.append((time.perf_counter() - t0) * 1000)
    return p > threshold


def simulate(min_silence_ms: int, gate_ms: int | None = None, hard_max_s: float = 12.0, gate_threshold: float = 0.5):
    """Returns cut positions (sample index where the decision is made)."""
    cuts, speaking, start, silence = [], False, 0, 0
    checked = False
    for w, p in enumerate(probs):
        t = w * WINDOW
        if not speaking:
            if p >= 0.5:
                speaking, start, silence, checked = True, t, 0, False
            continue
        silence = silence + 1 if p < 0.35 else 0
        if silence == 0:
            checked = False
        silence_ms = silence * WINDOW / SR * 1000
        if gate_ms and not checked and silence_ms >= gate_ms:
            checked = True
            if complete(stream[start:t], gate_threshold):
                cuts.append(t)
                speaking = False
                continue
        if silence_ms >= min_silence_ms or (t - start) / SR >= hard_max_s:
            cuts.append(t)
            speaking = False
    return cuts


def score(name: str, cuts: list[int]) -> None:
    ends_arr = np.array(ends)
    matched, delays, inside = set(), [], 0
    for c in cuts:
        # A cut belongs to the most recent sentence end before it, if within that sentence's gap.
        k = np.searchsorted(ends_arr, c, side="right") - 1
        if k >= 0 and k not in matched and c - ends_arr[k] < 1.6 * SR and (k + 1 >= len(ends) or c < ends_arr[k + 1]):
            nxt_start = ends_arr[k] + (len(parts[2 * k + 1]) if 2 * k + 1 < len(parts) else 0)
            if c <= nxt_start + 0.25 * SR:
                matched.add(k)
                delays.append((c - ends_arr[k]) / SR * 1000)
                continue
        inside += 1
    merged = len(ends) - len(matched)
    print(f"{name:34} fragments {inside:3d}   merged {merged:3d}   end→cut {np.mean(delays):4.0f} ms")


score("Silero 400 ms (earlier setting)", simulate(400))
score("Silero 700 ms (current)", simulate(700))
for thr in (0.5, 0.8, 0.9, 0.97):
    score(f"Smart Turn @300 ms p>{thr}, else 700", simulate(700, 300, gate_threshold=thr))
print(f"Smart Turn inference: {np.mean(turn_ms):.1f} ms per check (CPU)")
