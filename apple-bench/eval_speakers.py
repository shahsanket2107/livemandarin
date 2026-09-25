"""Speaker labels: CAM++ (current) vs ERes2NetV2 on ASCEND test sessions (real two-person
conversations with speaker IDs), using the server's own online clustering at several thresholds.

Metric per session: label accuracy under the best one-to-one mapping of predicted to true
speakers, and how many labels were created (ideal = number of real speakers).
Usage: ../server/.venv/bin/python eval_speakers.py
"""

import io
import sys
import time
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "server"))
from speakers import SpeakerTracker, load_embedder  # noqa: E402

MODELS = {"CAM++ (current)": ROOT.parent / "server/models/speaker-campplus-zh-en.onnx",
          "ERes2NetV2": ROOT / "data/eres2netv2.onnx"}

table = pq.read_table(ROOT / "data/ascend_test.parquet").to_pylist()
sessions = defaultdict(list)
for row in table:
    sessions[row["session_id"]].append(row)
sessions = {k: sorted(v, key=lambda r: r["path"]) for k, v in sessions.items()
            if len({r["original_speaker_id"] for r in v}) >= 2 and len(v) >= 20}


def decode(row) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def accuracy(pred: list, truth: list) -> float:
    pairs = [(p, t) for p, t in zip(pred, truth) if p is not None]
    if not pairs:
        return 0.0
    plabels, tlabels = sorted({p for p, _ in pairs}), sorted({t for _, t in pairs})
    counts = Counter(pairs)
    best = 0
    small, large = (plabels, tlabels) if len(plabels) <= len(tlabels) else (tlabels, plabels)
    for perm in permutations(large, len(small)):
        mapping = dict(zip(small, perm))
        if len(plabels) <= len(tlabels):
            score = sum(counts[(p, mapping[p])] for p in plabels)
        else:
            score = sum(counts[(mapping[t], t)] for t in tlabels)
        best = max(best, score)
    return best / len(pairs)


audio_cache = {k: [decode(r) for r in rows] for k, rows in sessions.items()}
print(f"{len(sessions)} sessions, {sum(len(v) for v in sessions.values())} utterances")
for name, path in MODELS.items():
    embed = load_embedder(path)
    t0 = time.perf_counter()
    embeddings = {k: [embed(a) if len(a) >= 12800 else None for a in audio_cache[k]] for k in sessions}
    ms = (time.perf_counter() - t0) / sum(len(v) for v in sessions.values()) * 1000
    for threshold in (0.55, 0.6, 0.65, 0.7, 0.75):
        accs, extra = [], []
        for k, rows in sessions.items():
            # Embeddings are precomputed; clips under 0.8 s stay unlabeled, as in the server.
            tracker = SpeakerTracker(lambda e: e, threshold=threshold, min_seconds=0.0)
            pred = [tracker.identify(e) if e is not None else None for e in embeddings[k]]
            truth = [r["original_speaker_id"] for r in rows]
            accs.append(accuracy(pred, truth))
            extra.append(len({p for p in pred if p is not None}) - len(set(truth)))
        print(f"{name:16} thr {threshold:.2f}: accuracy {100 * np.mean(accs):5.1f}%  "
              f"extra labels/session {np.mean(extra):+.1f}   ({ms:.0f} ms/utt)")
