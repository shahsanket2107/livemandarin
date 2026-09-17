"""Benchmark Qwen3-ASR variants on the labeled clips in bench/samples.

Usage: .venv/bin/python bench/bench_asr.py [--models Qwen/Qwen3-ASR-1.7B ...] [--bits 16 8 4]
"""

import argparse
import json
import re
import resource
import time
import unicodedata
from pathlib import Path

import mlx.core as mx
import soundfile as sf
from mlx_qwen3_asr import Session
from mlx_qwen3_asr.convert import quantize_model
from mlx_qwen3_asr.load_models import load_model

SAMPLES = Path(__file__).parent / "samples"
CONTEXT = "sprint review Kubernetes deployment Redis Grafana dashboard P99 alert React polyfill code splitting staging schema runbook rollback"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_session(model_id: str, bits: int) -> Session:
    model, _ = load_model(model_id, dtype=mx.float16)
    if bits in (4, 8):
        quantize_model(model, bits=bits)
    return Session(model=model, tokenizer_model=model_id)


def run(model_id: str, bits: int, use_context: bool) -> dict:
    session = load_session(model_id, bits)
    script = [json.loads(line) for line in (SAMPLES / "script.jsonl").open()]
    warm, _ = sf.read(SAMPLES / "s00.wav", dtype="float32")
    session.transcribe(warm)

    errors = chars = 0
    audio_sec = compute_sec = 0.0
    rows = []
    for i, item in enumerate(script):
        audio, sr = sf.read(SAMPLES / f"s{i:02d}.wav", dtype="float32")
        t0 = time.perf_counter()
        result = session.transcribe(audio, context=CONTEXT if use_context else "")
        dt = time.perf_counter() - t0
        ref, hyp = normalize(item["zh"]), normalize(result.text)
        errors += edit_distance(ref, hyp)
        chars += len(ref)
        audio_sec += len(audio) / sr
        compute_sec += dt
        rows.append({"ref": item["zh"], "hyp": result.text, "sec": round(dt, 3)})

    return {
        "model": model_id,
        "bits": bits,
        "context": use_context,
        "cer": round(100 * errors / chars, 2),
        "rtf": round(compute_sec / audio_sec, 4),
        "mean_utt_latency_s": round(compute_sec / len(script), 3),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"])
    parser.add_argument("--bits", nargs="+", type=int, default=[16, 8, 4])
    parser.add_argument("--out", default=str(Path(__file__).parent / "results_asr.json"))
    args = parser.parse_args()

    results = []
    for model_id in args.models:
        for bits in args.bits:
            for use_context in (False, True):
                mx.reset_peak_memory()
                r = run(model_id, bits, use_context)
                results.append(r)
                print(f"{model_id:24} {bits:>2}-bit ctx={use_context!s:5} CER={r['cer']:5}% "
                      f"RTF={r['rtf']} utt={r['mean_utt_latency_s']}s peak={r['peak_mem_gb']}GB", flush=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
