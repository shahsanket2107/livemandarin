import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter import SAMPLE_RATE, WINDOW, Segmenter, SegmenterConfig  # noqa: E402


def scripted_vad(pattern: list[tuple[float, float]]):
    """pattern: (seconds, prob) spans, converted to one prob per 32 ms window."""
    probs = []
    for seconds, prob in pattern:
        probs += [prob] * round(seconds * SAMPLE_RATE / WINDOW)
    it = iter(probs)
    return lambda _window: next(it, 0.0), sum(s for s, _ in pattern)


def run(pattern, config=None, chunk=1600):
    vad, total = scripted_vad(pattern)
    seg = Segmenter(vad, config or SegmenterConfig())
    audio = np.zeros(int(total * SAMPLE_RATE) + WINDOW, dtype=np.float32)
    events = []
    for i in range(0, len(audio), chunk):
        events += seg.feed(audio[i:i + chunk])
    events += seg.flush()
    return events


def utterance_lengths(events):
    return [round(len(e.audio) / SAMPLE_RATE, 1) for e in events if e.kind == "utterance"]


def test_two_sentences_split_on_pause():
    events = run([(0.5, 0.0), (2.0, 0.9), (0.8, 0.0), (1.5, 0.9), (1.0, 0.0)])
    assert [e.kind for e in events] == ["speech_start", "utterance", "speech_start", "utterance"]
    lengths = utterance_lengths(events)
    assert 2.0 <= lengths[0] <= 2.6 and 1.5 <= lengths[1] <= 2.1


def test_short_pause_does_not_split():
    events = run([(1.0, 0.9), (0.3, 0.0), (1.0, 0.9), (1.0, 0.0)])
    assert len(utterance_lengths(events)) == 1


def test_noise_blip_ignored():
    events = run([(1.0, 0.0), (0.1, 0.9), (1.0, 0.0)])
    assert utterance_lengths(events) == []


def test_long_monologue_cut_at_short_pause_after_soft_max():
    events = run([(7.0, 0.9), (0.25, 0.0), (3.0, 0.9), (1.0, 0.0)])
    lengths = utterance_lengths(events)
    assert len(lengths) == 2 and 7.0 <= lengths[0] <= 7.4


def test_hard_cap_without_pauses():
    events = run([(23.0, 0.9), (1.0, 0.0)], SegmenterConfig(hard_max_s=10.0))
    lengths = utterance_lengths(events)
    assert len(lengths) == 3 and all(length <= 10.1 for length in lengths)


def test_pre_roll_included():
    events = run([(1.0, 0.0), (1.0, 0.9), (1.0, 0.0)], SegmenterConfig(pre_roll_ms=200))
    assert utterance_lengths(events)[0] >= 1.2
