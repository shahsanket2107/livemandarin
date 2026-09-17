"""Hy-MT2 translation: prompt construction (official templates) and a streaming Ollama client."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

log = logging.getLogger("mt")


@dataclass(frozen=True)
class Direction:
    """One translation direction with Hy-MT2's official prompt wording for it."""

    name: str            # "zh-en" | "en-zh"
    instruction: str
    term_header: str
    term_line: str       # format with zh=…, en=…

    def term_applies(self, term: "GlossaryTerm", source: str) -> bool:
        return term.zh in source if self.name == "zh-en" else term.en.lower() in source.lower()


ZH_EN = Direction("zh-en", "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：",
                  "参考下面的翻译：", "{zh} 翻译成 {en}")
EN_ZH = Direction("en-zh", "Translate the following text into Chinese. Note that you must ONLY output the "
                  "translated result without any additional explanation:",
                  "Reference the following translations:", "{en} translates to {zh}")
DIRECTIONS = {d.name: d for d in (ZH_EN, EN_ZH)}


@dataclass(frozen=True)
class GlossaryTerm:
    zh: str
    en: str


@dataclass(frozen=True)
class Turn:
    src: str
    dst: str
    direction: Direction = ZH_EN


@dataclass(frozen=True)
class ChatFormat:
    """Raw chat markup for one Hy-MT2 model family (from each repo's chat_template.jinja).

    Ollama's auto-converted template for the 1.8B GGUF is broken, so prompts are rendered
    here and sent with raw=True.
    """

    name: str
    bos: str            # emitted once at the very start
    user_open: str
    user_close: str     # also opens the assistant reply
    assistant_close: str
    stop: tuple[str, ...]


HY_DENSE = ChatFormat(  # Hy-MT2-1.8B
    name="hy_dense",
    bos="<｜hy_begin▁of▁sentence｜>",
    user_open="<｜hy_User｜>",
    user_close="<｜hy_Assistant｜>",
    assistant_close="<｜hy_place▁holder▁no▁2｜>",
    stop=("<｜hy_place▁holder▁no▁2｜>", "<｜hy_place▁holder▁no▁8｜>", "<｜hy_User｜>"),
)
HUNYUAN = ChatFormat(  # Hy-MT2-7B
    name="hunyuan",
    bos="",
    user_open="<|startoftext|>",
    user_close="<|extra_0|>",
    assistant_close="<|eos|>",
    stop=("<|eos|>", "<|startoftext|>", "<|extra_0|>"),
)
FORMATS_BY_BOS = {"<｜hy_begin▁of▁sentence｜>": HY_DENSE, "<|startoftext|>": HUNYUAN}


def build_prompt(source: str, terms: list[GlossaryTerm], direction: Direction = ZH_EN) -> str:
    """Hy-MT2's terminology + default translation template for one sentence.

    Only glossary terms that actually appear in the source are included, which keeps the
    prompt short (fast prefill) and avoids nudging the model with irrelevant terms.
    """
    relevant = [t for t in terms if direction.term_applies(t, source)]
    prefix = ""
    if relevant:
        lines = "\n".join(direction.term_line.format(zh=t.zh, en=t.en) for t in relevant)
        prefix = f"{direction.term_header}\n{lines}\n\n"
    return f"{prefix}{direction.instruction}\n\n{source}"


def format_chat(fmt: ChatFormat, source: str, history: list[Turn], terms: list[GlossaryTerm],
                direction: Direction = ZH_EN) -> str:
    """Render the raw chat prompt.

    Earlier sentences and their translations are given as previous user/assistant turns.
    This supplies conversational context (pronouns, topic, terminology consistency) without
    the model re-translating the context, which the "background information" template
    encouraged in benchmarks.
    """
    parts = [fmt.bos]
    for turn in history:
        parts.append(f"{fmt.user_open}{build_prompt(turn.src, terms, turn.direction)}{fmt.user_close}"
                     f"{turn.dst}{fmt.assistant_close}")
    parts.append(f"{fmt.user_open}{build_prompt(source, terms, direction)}{fmt.user_close}")
    return "".join(parts)


class Translator:
    """Streams translations from Ollama's raw generate endpoint."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.2, num_ctx: int = 2048):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60)
        self.model = model
        self.format: ChatFormat | None = None
        self._options = {"temperature": temperature, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05,
                         "num_ctx": num_ctx, "num_predict": 256}

    async def load(self) -> None:
        """Detect the chat format from the GGUF's BOS token and warm the model into memory."""
        resp = await self._client.post("/api/show", json={"model": self.model, "verbose": True})
        resp.raise_for_status()
        info = resp.json()["model_info"]
        bos = info["tokenizer.ggml.tokens"][info["tokenizer.ggml.bos_token_id"]]
        if bos not in FORMATS_BY_BOS:
            raise RuntimeError(f"{self.model}: unknown chat format (BOS token {bos!r})")
        self.format = FORMATS_BY_BOS[bos]
        self._options["stop"] = list(self.format.stop)
        text = await self.translate("你好，我们开始吧。", [], [])
        log.info("MT %s (%s format) ready: %r", self.model, self.format.name, text)

    async def stream(self, source: str, history: list[Turn], terms: list[GlossaryTerm],
                     direction: Direction = ZH_EN) -> AsyncIterator[str]:
        """Yield the translation incrementally, one generated piece at a time."""
        assert self.format is not None, "call load() first"
        body = {
            "model": self.model,
            "prompt": format_chat(self.format, source, history, terms, direction),
            "raw": True,
            "stream": True,
            "keep_alive": -1,
            "options": self._options,
        }
        async with self._client.stream("POST", "/api/generate", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    piece = json.loads(line).get("response", "")
                    if piece:
                        yield piece

    async def translate(self, source: str, history: list[Turn], terms: list[GlossaryTerm],
                        direction: Direction = ZH_EN) -> str:
        return "".join([piece async for piece in self.stream(source, history, terms, direction)]).strip()

    async def aclose(self) -> None:
        await self._client.aclose()
