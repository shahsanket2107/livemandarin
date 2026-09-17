import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speakers import SAMPLE_RATE, SpeakerTracker  # noqa: E402

VOICES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]), "c": np.array([0.0, 0.0, 1.0])}


def fake_embed(audio: np.ndarray) -> np.ndarray:
    """The audio's first sample encodes which voice it is; add a little noise."""
    base = VOICES[chr(int(audio[0]))]
    return base + np.random.default_rng(int(audio[1])).normal(0, 0.15, 3)


def clip(voice: str, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    audio[0], audio[1] = ord(voice), seed
    return audio


def test_same_voice_keeps_label_and_new_voice_gets_new_label():
    tracker = SpeakerTracker(fake_embed, threshold=0.7)
    labels = [tracker.identify(clip(v, seed=i)) for i, v in enumerate("aabacab")]
    assert labels == [0, 0, 1, 0, 2, 0, 1]


def test_short_utterance_is_unlabeled():
    tracker = SpeakerTracker(fake_embed, threshold=0.7, min_seconds=0.8)
    assert tracker.identify(clip("a", seconds=0.5)) is None
    assert tracker.identify(clip("a", seconds=1.0)) == 0


def test_max_speakers_caps_new_labels():
    tracker = SpeakerTracker(fake_embed, threshold=0.99, max_speakers=2)
    assert [tracker.identify(clip(v, seed=i)) for i, v in enumerate("abc")][:2] == [0, 1]
    assert tracker.identify(clip("c", seed=9)) in (0, 1)
