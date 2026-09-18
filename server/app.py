"""Local caption server: 16 kHz PCM over WebSocket in, English captions (JSON) out.

Protocol (ws://127.0.0.1:8765/ws):
  client -> server   binary frames of little-endian Int16 mono PCM at 16 kHz (any size)
                     text  {"type":"flush"}  end the current utterance now (e.g. on stop)
                     text  {"type":"config", "mode": "zh-en"|"en-zh"|"auto"}  translation direction
  server -> client   {"type":"ready", "asr", "mt"}
                     {"type":"listening"}                       someone started speaking
                     {"type":"processing", "id"}                they paused; transcribing
                     {"type":"partial", "id", "dst", "dir", "speaker"}  translation so far (streamed)
                     {"type":"final", "id", "src", "dst", "dir", "speaker", "t_ms"} caption complete
                     (dir: "zh-en"|"en-zh"; speaker: 0-based label by voice, consistent within
                     the session, or null)
                     {"type":"drop", "id"}                      nothing intelligible was said
GET /health returns 200 once models are loaded (the app waits for it).
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from collections import deque
from http import HTTPStatus
from pathlib import Path

import numpy as np
import yaml
from websockets.asyncio.server import Request, ServerConnection, serve

from asr import Recognizer
from mt import EN_ZH, ZH_EN, Direction, GlossaryTerm, Translator, Turn
from segmenter import Segmenter, SegmenterConfig, SileroVAD
from speakers import SpeakerTracker, load_embedder

ROOT = Path(__file__).resolve().parent
SERVER_ID = "livemandarin/1"  # reported by /health so the app can spot a foreign or outdated server
PARTIAL_INTERVAL = 0.05  # seconds between streamed caption updates
log = logging.getLogger("app")


class Glossary:
    """glossary.yaml, re-read whenever its mtime changes."""

    def __init__(self, path: Path):
        self.path = path
        self._mtime = 0.0
        self.hints = ""
        self.terms: list[GlossaryTerm] = []
        self.refresh()

    def refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        data = yaml.safe_load(self.path.read_text()) or {}
        self.hints = " ".join(str(h) for h in data.get("asr_hints") or [])
        self.terms = [GlossaryTerm(str(t["zh"]), str(t["en"])) for t in data.get("terms") or []]
        log.info("glossary: %d hints, %d terms", len(self.hints.split()), len(self.terms))


CHINESE_LANGUAGES = {"chinese", "cantonese", "mandarin", "zh", "yue"}
MODES = {"zh-en": {ZH_EN}, "en-zh": {EN_ZH}, "auto": {ZH_EN, EN_ZH}}


def has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def is_chinese(language: str, text: str) -> bool:
    """Chinese in any form: the recognizer's language label, or Chinese characters in the text
    (covers mixed sentences the detector may label oddly)."""
    return language.strip().lower() in CHINESE_LANGUAGES or has_cjk(text)


FILLER_CHARS = set("嗯呃啊哦噢欸诶唉呀嘛吧哈额嘿唔哼")
FILLER_WORDS = {"uh", "um", "mm", "hmm", "mmhmm", "mm-hmm", "uh-huh", "ah", "oh", "eh", "hm", "yeah", "yep", "ok", "okay"}


def is_filler(text: str) -> bool:
    """Backchannel noises ("嗯嗯", "呃", "uh-huh") that would only clutter the captions."""
    core = "".join(ch for ch in text if ch.isalnum() or ch == "-").lower()
    if not core:
        return True
    if all(ch in FILLER_CHARS for ch in core):
        return True
    return core in FILLER_WORDS


def direction_for(language: str, text: str) -> Direction | None:
    if is_chinese(language, text):
        return ZH_EN
    if language.strip().lower() == "english":
        return EN_ZH
    return None


class Server:
    def __init__(self, config: dict):
        self.config = config
        asr, mt = config["asr"], config["mt"]
        self.recognizer = Recognizer(asr["model"], asr.get("language"))
        self.translator = Translator(mt["ollama_url"], mt["model"], mt.get("temperature", 0.2))
        self.glossary = Glossary(ROOT / "glossary.yaml")
        self.history_size = mt.get("history_size", 3)
        speakers = config.get("speakers", {})
        self.speaker_embed = load_embedder(ROOT / speakers["model"]) if speakers.get("enabled") else None
        self.speaker_config = speakers
        self.log_text = bool(config["server"].get("log_text", False))
        self.default_mode = config.get("caption", {}).get("mode", "zh-en")
        if self.default_mode not in MODES:
            raise ValueError(f"caption.mode must be one of {sorted(MODES)}")
        self.ready = False

    async def load(self) -> None:
        self.recognizer.start()
        await self.translator.load()
        await asyncio.to_thread(self.recognizer.ready.wait)
        self.ready = True

    async def process_request(self, connection: ServerConnection, request: Request):
        if request.path == "/health":
            body = ("ready" if self.ready else "loading") + f" {SERVER_ID}\n"
            return connection.respond(HTTPStatus.OK if self.ready else HTTPStatus.SERVICE_UNAVAILABLE, body)
        origin = request.headers.get("Origin", "")
        if origin and origin != "app://livemandarin":  # never accept connections from web pages
            return connection.respond(HTTPStatus.FORBIDDEN, "forbidden\n")
        return None

    async def handle(self, ws: ServerConnection) -> None:
        log.info("client connected (%s)", ws.request.headers.get("Origin", "no origin"))
        session = Session(self, ws)
        try:
            await session.run()
        finally:
            session.close()
            log.info("client disconnected")


class Session:
    """One connected client: audio in → segmenter → ASR → MT → captions out."""

    def __init__(self, server: Server, ws: ServerConnection):
        self.server = server
        self.ws = ws
        self.segmenter = Segmenter(SileroVAD(), SegmenterConfig(**server.config.get("segmenter", {})))
        self.history: deque[Turn] = deque(maxlen=server.history_size)
        self.mode = server.default_mode
        cfg = server.speaker_config
        self.speakers = SpeakerTracker(server.speaker_embed, cfg.get("threshold", 0.7), cfg.get("min_seconds", 0.8),
                                       cfg.get("max_speakers", 8)) if server.speaker_embed else None
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.text_queue: asyncio.Queue = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.next_id = 1
        self._frames = 0
        self._peak = 0.0

    async def send(self, message: dict) -> None:
        await self.ws.send(json.dumps(message, ensure_ascii=False))

    async def run(self) -> None:
        self.tasks = [asyncio.create_task(self.asr_worker()), asyncio.create_task(self.mt_worker())]
        await self.send({"type": "ready", "asr": self.server.recognizer.model_id, "mt": self.server.translator.model})
        async for message in self.ws:
            if isinstance(message, bytes):
                samples = np.frombuffer(message, dtype="<i2").astype(np.float32) / 32768.0
                self._report_audio(samples)
                await self.dispatch(self.segmenter.feed(samples))
            else:
                command = json.loads(message)
                if command.get("type") == "flush":
                    await self.dispatch(self.segmenter.flush())
                elif command.get("type") == "config" and command.get("mode") in MODES:
                    self.mode = command["mode"]
                    log.info("translation mode: %s", self.mode)

    def _report_audio(self, samples: np.ndarray) -> None:
        """Log incoming audio level every ~10 s so 'no captions' is diagnosable."""
        self._frames += 1
        self._peak = max(self._peak, float(np.abs(samples).max(initial=0.0)))
        if self._frames % 100 == 0:
            level = "silence" if self._peak < 0.005 else f"peak {self._peak:.2f}"
            log.info("audio: %d frames received, last 10 s: %s", self._frames, level)
            self._peak = 0.0

    async def dispatch(self, events) -> None:
        for event in events:
            if event.kind == "speech_start":
                self.server.recognizer.set_speaking(True)
                await self.send({"type": "listening"})
            else:
                self.server.recognizer.set_speaking(False)
                uid, self.next_id = self.next_id, self.next_id + 1
                self.audio_queue.put_nowait((uid, event.audio, time.perf_counter()))
                await self.send({"type": "processing", "id": uid})

    async def asr_worker(self) -> None:
        while True:
            uid, audio, t_end = await self.audio_queue.get()
            self.server.glossary.refresh()
            # Voice print runs on the CPU while the recognizer uses the GPU.
            speaker_job = asyncio.to_thread(self.speakers.identify, audio) if self.speakers else None
            transcript = await self.server.recognizer.transcribe(audio, self.server.glossary.hints)
            speaker = await speaker_job if speaker_job else None
            if transcript.text and is_filler(transcript.text):
                await self.send({"type": "drop", "id": uid})
                continue
            direction = direction_for(transcript.language, transcript.text) if transcript.text else None
            if direction is None and transcript.text and ZH_EN in MODES[self.mode]:
                # Fast or unclear Mandarin is sometimes labelled as another language; ask for Chinese explicitly.
                retry = await self.server.recognizer.transcribe(audio, self.server.glossary.hints, language="Chinese")
                if has_cjk(retry.text):
                    log.info("#%d relabelled %s -> Chinese on retry", uid, transcript.language or "unknown")
                    transcript, direction = retry, ZH_EN
            if direction in MODES[self.mode]:
                self.text_queue.put_nowait((uid, len(audio) / 16000, t_end, transcript, speaker, direction))
            else:
                if transcript.text:
                    log.info("#%d skipped: %s speech (mode %s)", uid, transcript.language or "unknown", self.mode)
                await self.send({"type": "drop", "id": uid})

    async def mt_worker(self) -> None:
        while True:
            uid, audio_s, t_end, transcript, speaker, direction = await self.text_queue.get()
            src = transcript.text
            t0 = time.perf_counter()
            dst = await self.stream_translation(uid, src, speaker, direction)
            self.history.append(Turn(src, dst, direction))
            t_ms = round((time.perf_counter() - t_end) * 1000)
            log.info("#%d %.1fs audio | %s | spk %s | asr %.2fs | mt %.2fs | %d ms%s",
                     uid, audio_s, transcript.language or "?", speaker, transcript.seconds, time.perf_counter() - t0, t_ms,
                     f" | {src} -> {dst}" if self.server.log_text else "")
            await self.send({"type": "final", "id": uid, "src": src, "dst": dst, "dir": direction.name,
                             "speaker": speaker, "t_ms": t_ms})

    async def stream_translation(self, uid: int, src: str, speaker: int | None, direction: Direction) -> str:
        """Stream partial captions as the translation is generated; return the full text."""
        pieces: list[str] = []
        last_sent = 0.0
        async for piece in self.server.translator.stream(src, list(self.history), self.server.glossary.terms,
                                                         direction):
            pieces.append(piece)
            now = time.perf_counter()
            if now - last_sent >= PARTIAL_INTERVAL:
                await self.send({"type": "partial", "id": uid, "dst": "".join(pieces).strip(),
                                 "dir": direction.name, "speaker": speaker})
                last_sent = now
        return "".join(pieces).strip()

    def close(self) -> None:
        for task in self.tasks:
            task.cancel()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    server = Server(config)
    host, port = config["server"]["host"], config["server"]["port"]
    async with serve(server.handle, host, port, process_request=server.process_request,
                     max_size=2**22, compression=None):
        log.info("listening on ws://%s:%d/ws — loading models…", host, port)
        await server.load()
        log.info("ready")
        await asyncio.Future()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # clean exit also stops the ASR child process
    asyncio.run(main())
