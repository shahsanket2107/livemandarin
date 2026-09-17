"""Synthesize the labeled test clips from script.jsonl with macOS Chinese voices.

Usage: .venv/bin/python bench/make_samples.py   (needs ffmpeg: brew install ffmpeg)
Writes s00.wav … sNN.wav (16 kHz mono) plus meeting.wav, the clips joined with 0.7 s gaps.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLES = Path(__file__).parent / "samples"
GAP = np.zeros(int(0.7 * 16000), dtype=np.float32)


def main() -> None:
    script = [json.loads(line) for line in (SAMPLES / "script.jsonl").open()]
    parts = []
    for i, item in enumerate(script):
        aiff, wav = SAMPLES / f"s{i:02d}.aiff", SAMPLES / f"s{i:02d}.wav"
        subprocess.run(["say", "-v", item["voice"], "-o", str(aiff), item["zh"]], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)],
                       check=True)
        aiff.unlink()
        audio, _ = sf.read(wav, dtype="float32")
        parts += [audio, GAP]
    sf.write(SAMPLES / "meeting.wav", np.concatenate(parts), 16000)
    print(f"{len(script)} clips, meeting.wav = {sum(len(p) for p in parts) / 16000:.1f} s")


if __name__ == "__main__":
    main()
