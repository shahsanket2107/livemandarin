"""Voice-activity segmentation of a 16 kHz mono stream into utterances.

An utterance ends after `min_silence_ms` of non-speech. Long monologues are cut earlier so
captions keep flowing: past `soft_max_s` a much shorter pause is enough, and at `hard_max_s`
the utterance is cut unconditionally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 512  # samples per VAD decision (32 ms), fixed by Silero


class SileroVAD:
    """Minimal onnxruntime wrapper around the Silero VAD model (no torch at runtime)."""

    CONTEXT = 64

    def __init__(self, model_path: str | Path | None = None):
        import onnxruntime as ort

        if model_path is None:
            import importlib.resources

            model_path = importlib.resources.files("silero_vad") / "data" / "silero_vad.onnx"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self._context, window.reshape(1, WINDOW)], axis=1)
        prob, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self._context = x[:, -self.CONTEXT:]
        return float(prob.item())


@dataclass
class SegmenterConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_silence_ms: int = 500
    soft_max_s: float = 6.0
    soft_silence_ms: int = 160
    hard_max_s: float = 10.0
    min_speech_ms: int = 250
    pre_roll_ms: int = 200


@dataclass
class Event:
    kind: str  # "speech_start" | "utterance"
    audio: np.ndarray | None = None
    end_sample: int = 0  # stream position (samples) where the utterance ended


@dataclass
class Segmenter:
    vad: Callable[[np.ndarray], float]
    config: SegmenterConfig = field(default_factory=SegmenterConfig)

    def __post_init__(self) -> None:
        c = self.config
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, c.pre_roll_ms * SAMPLE_RATE // 1000 // WINDOW))
        self._speech: list[np.ndarray] = []
        self._speaking = False
        self._silence_windows = 0
        self._voiced_windows = 0
        self._position = 0

    def _ms_to_windows(self, ms: float) -> int:
        return max(1, round(ms * SAMPLE_RATE / 1000 / WINDOW))

    def feed(self, samples: np.ndarray) -> list[Event]:
        """Feed float32 samples in [-1, 1]; returns events produced by this chunk."""
        events: list[Event] = []
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = len(buf) // WINDOW
        for i in range(n):
            events.extend(self._step(buf[i * WINDOW:(i + 1) * WINDOW]))
        self._pending = buf[n * WINDOW:]
        return events

    def flush(self) -> list[Event]:
        return self._finish() if self._speaking else []

    def _step(self, window: np.ndarray) -> list[Event]:
        c = self.config
        prob = self.vad(window)
        self._position += WINDOW

        if not self._speaking:
            if prob >= c.start_threshold:
                self._speaking = True
                self._speech = list(self._pre_roll) + [window]
                self._pre_roll.clear()
                self._silence_windows = 0
                self._voiced_windows = 1
                return [Event("speech_start")]
            self._pre_roll.append(window)
            return []

        self._speech.append(window)
        if prob < c.end_threshold:
            self._silence_windows += 1
        else:
            self._silence_windows = 0
            self._voiced_windows += 1

        duration_s = len(self._speech) * WINDOW / SAMPLE_RATE
        needed = c.min_silence_ms if duration_s < c.soft_max_s else c.soft_silence_ms
        if self._silence_windows >= self._ms_to_windows(needed) or duration_s >= c.hard_max_s:
            return self._finish()
        return []

    def _finish(self) -> list[Event]:
        c = self.config
        # Trim most of the trailing silence but keep a little so final syllables aren't clipped.
        keep_tail = self._ms_to_windows(120)
        trim = max(0, self._silence_windows - keep_tail)
        windows = self._speech[:len(self._speech) - trim] if trim else self._speech
        voiced_ok = self._voiced_windows >= self._ms_to_windows(c.min_speech_ms)
        still_speaking = self._silence_windows == 0  # hard cut mid-speech

        self._speech = []
        self._silence_windows = 0
        self._voiced_windows = 0
        self._speaking = still_speaking
        if still_speaking:
            self._voiced_windows = 1
        if not voiced_ok:
            return []
        return [Event("utterance", np.concatenate(windows), self._position)]
