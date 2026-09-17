"""Speaker labels: a CAM++ voice-embedding model (sherpa-onnx, CPU, ~40 ms per utterance) plus
online clustering, so the same voice keeps the same label for the whole session.

Labels are indexes (0, 1, 2 …) in order of first appearance; the app shows them as
"Speaker 1", "Speaker 2" …. Utterances shorter than `min_seconds` get no label — too little
audio for a reliable voice print.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_embedder(model_path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=1)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(audio: np.ndarray) -> np.ndarray:
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        return np.asarray(extractor.compute(stream), dtype=np.float32)

    return embed


class SpeakerTracker:
    def __init__(self, embed: Callable[[np.ndarray], np.ndarray], threshold: float = 0.7,
                 min_seconds: float = 0.8, max_speakers: int = 8):
        self._embed = embed
        self.threshold = threshold
        self.min_samples = int(min_seconds * SAMPLE_RATE)
        self.max_speakers = max_speakers
        self._centroids: list[np.ndarray] = []  # unnormalized sums of unit embeddings

    def identify(self, audio: np.ndarray) -> int | None:
        if len(audio) < self.min_samples:
            return None
        e = self._embed(audio)
        e = e / (np.linalg.norm(e) + 1e-8)
        if self._centroids:
            sims = [float(e @ c / (np.linalg.norm(c) + 1e-8)) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold or len(self._centroids) >= self.max_speakers:
                self._centroids[best] = self._centroids[best] + e
                return best
        self._centroids.append(e.copy())
        return len(self._centroids) - 1

    def reset(self) -> None:
        self._centroids.clear()
