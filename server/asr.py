"""Qwen3-ASR (MLX) in a dedicated child process.

Why a process: the same transcription measured ~0.85 s in a plain process but 1.1–1.9 s when
the model shared a process with the asyncio/WebSocket thread (interpreter-lock and macOS
thread-scheduling effects). A separate process keeps the model on its own main thread.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection

import numpy as np

log = logging.getLogger("asr")
QOS_CLASS_USER_INTERACTIVE = 0x21


@dataclass
class Transcript:
    text: str
    language: str
    seconds: float


def _worker(model_id: str, language: str | None, conn: Connection) -> None:
    """Child process: load the model, then serve transcription requests over `conn`.

    While someone is speaking the loop stays lightly busy (small CPU and GPU matmuls every
    ~50 ms); otherwise macOS parks the core and clocks the GPU down during the pause, and the
    next transcription measured 40–60% slower. Nothing runs during silence.
    """
    import mlx.core as mx
    from mlx_qwen3_asr import Session

    ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    session = Session(model=model_id)
    session.transcribe(np.zeros(16000, dtype=np.float32))  # compile kernels
    conn.send(("ready", None))

    gpu_warm = mx.ones((1024, 1024), dtype=mx.float16)
    cpu_warm = np.random.rand(64, 64)
    speaking = False
    while True:
        if not conn.poll(0.01 if speaking else 0.1):
            if speaking:
                mx.eval(gpu_warm @ gpu_warm)
                deadline = time.perf_counter() + 0.01
                while time.perf_counter() < deadline:
                    cpu_warm = cpu_warm @ cpu_warm / 64 + 1e-3
            continue
        try:
            kind, payload = conn.recv()
        except EOFError:  # parent went away
            return
        if kind == "speaking":
            speaking = payload
        elif kind == "transcribe":
            uid, audio, context = payload
            t0 = time.perf_counter()
            result = session.transcribe(audio, context=context, language=language)
            conn.send(("result", (uid, result.text.strip(), result.language or "", time.perf_counter() - t0)))


class Recognizer:
    def __init__(self, model_id: str, language: str | None = None):
        self.model_id = model_id
        self.language = language
        self.ready = threading.Event()
        self._conn, child_conn = mp.get_context("spawn").Pipe()
        self._process = mp.get_context("spawn").Process(
            target=_worker, args=(model_id, language, child_conn), name="asr", daemon=True)
        self._pending: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        t0 = time.perf_counter()
        self._process.start()
        threading.Thread(target=self._reader, args=(t0,), name="asr-reader", daemon=True).start()

    def _reader(self, t0: float) -> None:
        while True:
            try:
                kind, payload = self._conn.recv()
            except EOFError:
                log.error("ASR process exited")
                return
            if kind == "ready":
                self.ready.set()
                log.info("ASR %s ready in %.1fs", self.model_id, time.perf_counter() - t0)
            elif kind == "result":
                uid, text, lang, seconds = payload
                loop, future = self._pending.pop(uid)
                loop.call_soon_threadsafe(future.set_result, Transcript(text, lang, seconds))

    def set_speaking(self, speaking: bool) -> None:
        with self._lock:
            self._conn.send(("speaking", speaking))

    async def transcribe(self, audio: np.ndarray, context: str = "") -> Transcript:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        with self._lock:
            uid, self._next_id = self._next_id, self._next_id + 1
            self._pending[uid] = (loop, future)
            self._conn.send(("transcribe", (uid, audio, context)))
        return await future
