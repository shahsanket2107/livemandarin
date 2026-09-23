"""How does Mandarin recognition degrade under call-like conditions, and what helps?

Applies Meet/Zoom-like degradations to the synthetic test clips (Opus codec at call bitrate,
faster speech, background noise, all three) and measures CER for each, with and without
passing the previous sentence's transcript as recognizer context.

Usage: .venv/bin/python bench/bench_robustness.py   (needs ffmpeg)
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_asr import SAMPLES, edit_distance, normalize  # noqa: E402

MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
CONDITIONS = {
    "clean": [],
    "opus-24k": ["-c:a", "libopus", "-b:a", "24k"],
    "fast-1.3x": ["-filter:a", "atempo=1.3"],
    "noise": None,  # handled in numpy
    "call (opus+fast+noise)": ["-filter:a", "atempo=1.3", "-c:a", "libopus", "-b:a", "24k"],
}


def degrade(src: Path, args: list[str] | None, noise: bool, tmp: Path) -> np.ndarray:
    audio, _ = sf.read(src, dtype="float32")
    if args:
        mid = tmp / (src.stem + ".ogg" if "libopus" in args else src.stem + "_f.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), *args, str(mid)], check=True)
        out = tmp / (src.stem + "_d.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mid), "-ar", "16000", "-ac", "1", str(out)],
                       check=True)
        audio, _ = sf.read(out, dtype="float32")
    if noise:
        rng = np.random.default_rng(0)
        audio = audio + rng.normal(0, 0.01, len(audio)).astype(np.float32)  # ~20 dB SNR
    return audio


def main() -> None:
    from mlx_qwen3_asr import Session

    script = [json.loads(line) for line in (SAMPLES / "script.jsonl").open()]
    hints = " ".join(yaml.safe_load((SAMPLES.parents[1] / "glossary.yaml").read_text())["asr_hints"])
    session = Session(model=MODEL)
    session.transcribe(np.zeros(16000, dtype=np.float32))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"{'condition':24} {'hints only':>11} {'+prev sentence':>15}   sec/utt")
        for name, args in CONDITIONS.items():
            noise = name in ("noise", "call (opus+fast+noise)")
            clips = [degrade(SAMPLES / f"s{i:02d}.wav", args, noise, tmp) for i in range(len(script))]
            results = []
            for use_prev in (False, True):
                errors = chars = 0
                prev = ""
                t0 = time.perf_counter()
                for item, clip in zip(script, clips):
                    context = f"{hints} {prev}" if use_prev and prev else hints
                    text = session.transcribe(clip, context=context).text
                    ref, hyp = normalize(item["zh"]), normalize(text)
                    errors += edit_distance(ref, hyp)
                    chars += len(ref)
                    prev = text
                results.append((100 * errors / chars, (time.perf_counter() - t0) / len(script)))
            print(f"{name:24} {results[0][0]:>10.1f}% {results[1][0]:>14.1f}%   {results[1][1]:.2f}")


if __name__ == "__main__":
    main()
