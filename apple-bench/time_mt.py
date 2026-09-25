"""Clean latency pass: 40 FLEURS sentences per finalist, nothing else running, greedy decoding."""
import json, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mt import CANDIDATES, dataset, make_prompt
rows = [r for r in dataset() if r["set"] == "fleurs"][:40]
for name in sys.argv[1:]:
    model, tok = load(CANDIDATES[name]); s = make_sampler(temp=0.0)
    generate(model, tok, prompt=make_prompt(name, tok, "你好。", False), max_tokens=8, sampler=s)
    times = []
    for r in rows:
        t0 = time.perf_counter(); generate(model, tok, prompt=make_prompt(name, tok, r["zh"], False), max_tokens=160, sampler=s); times.append(time.perf_counter() - t0)
    times.sort(); print(f"{name:24} median {times[len(times)//2]:.2f}s  p90 {times[int(len(times)*.9)]:.2f}s", flush=True)
    del model; mx.clear_cache()
