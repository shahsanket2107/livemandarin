"""End-to-end smoke test: stream a WAV to the running server as the extension would.

Usage: .venv/bin/python bench/stream_client.py [bench/samples/meeting.wav] [--speed 1.0]
Sends 100 ms Int16 frames in real time (or faster with --speed) and prints each caption with
the server-measured latency from end of speech to (a) first streamed words and (b) the full
caption.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

FRAME = 1600  # 100 ms at 16 kHz


def percentile(values: list[int], p: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", default=str(Path(__file__).parent / "samples" / "meeting.wav"))
    parser.add_argument("--speed", type=float, default=1.0, help=">1 streams faster than real time")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--mode", default=None, help="zh-en | en-zh | auto (server default if omitted)")
    args = parser.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000, "resample to 16 kHz first"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

    first_words: dict[int, float] = {}
    processing_at: dict[int, float] = {}
    first_ms: list[int] = []
    final_ms: list[int] = []

    async with connect(args.url, origin="app://meet-captions") as ws:
        print(json.loads(await ws.recv()))
        if args.mode:
            await ws.send(json.dumps({"type": "config", "mode": args.mode}))

        async def sender() -> None:
            t0 = time.perf_counter()
            for i in range(0, len(pcm), FRAME * 2):
                await ws.send(pcm[i:i + FRAME * 2])
                target = t0 + (i // 2 + FRAME) / 16000 / args.speed
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
            await ws.send(json.dumps({"type": "flush"}))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=8.0))
                now = time.perf_counter()
                if msg["type"] == "processing":
                    processing_at[msg["id"]] = now
                elif msg["type"] == "partial" and msg["id"] not in first_words:
                    first_words[msg["id"]] = now
                elif msg["type"] == "final":
                    uid = msg["id"]
                    first = round((first_words.get(uid, now) - processing_at[uid]) * 1000)
                    first_ms.append(first)
                    final_ms.append(msg["t_ms"])
                    who = f"S{msg['speaker'] + 1}" if msg.get("speaker") is not None else "--"
                    print(f"[first {first:4d} ms | full {msg['t_ms']:4d} ms] {who} {msg['dst']}\n{'':31}{msg['src']}")
        except asyncio.TimeoutError:
            pass
        await send_task

    if not final_ms:
        print("no captions received", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(final_ms)} captions | first words p50 {percentile(first_ms, .5)} ms, p95 {percentile(first_ms, .95)} ms"
          f" | full caption p50 {percentile(final_ms, .5)} ms, p95 {percentile(final_ms, .95)} ms")


if __name__ == "__main__":
    asyncio.run(main())
