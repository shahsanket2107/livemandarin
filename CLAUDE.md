# CLAUDE.md — LiveMandarin

Local, private live-translation captions for calls on a Mac: a Swift menu-bar app captures
system audio, a Python server runs Silero VAD → Qwen3-ASR-1.7B → CAM++ speaker embeddings →
Hy-MT2-1.8B (via Ollama), and a floating panel shows the captions. README.md is the user-facing
overview; this file is for you, Claude, working in this repo.

## If the user asks you to set this up on their Mac

Do it for them end to end — clone, install every prerequisite (Homebrew packages, Ollama,
Python, the three models, the signing certificate), build and install the app, verify it, and
tailor the glossary. They should not need to write code or run commands themselves except
where macOS requires their own password or click (Homebrew's installer, the Command Line Tools
dialog, permission prompts). Work through this list in order and verify each step instead of
assuming it worked.

Requirements you are responsible for satisfying: Apple Silicon Mac, macOS ≥ 14, ≥ 8 GB free
disk, Homebrew, Command Line Tools with a working `swiftc`, Ollama (`brew install ollama`),
Python 3.13 (`brew install python@3.13`), ffmpeg, the Python packages in
`server/requirements.txt`, the Hy-MT2 model in Ollama, the Qwen3-ASR model in the Hugging Face
cache, the CAM++ model in `server/models/`. `scripts/setup.sh` does all of this; run it rather
than reproducing the steps by hand, and re-run it after fixing whatever it stopped on.

1. **Preflight (read-only):** `uname -m` must be `arm64`; `sw_vers -productVersion` ≥ 14;
   `df -h ~` has ≥ 8 GB free; `brew --version` works; `xcode-select -p` and `swiftc -version`
   work. If Homebrew is missing, give them the one-line installer from https://brew.sh (needs
   their password — they run it). If Command Line Tools are missing, they run
   `xcode-select --install` and click Install.
2. **Run `scripts/setup.sh`** (≈10 min, ~5 GB download). It is idempotent. Watch for:
   - *"SDK is not supported by the compiler"* from `swiftc`: the Command Line Tools are stale.
     Fix: `sudo mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools.old`
     (they type the password) then `xcode-select --install`; re-run setup.
   - The build uses `swiftc` directly, not SwiftPM, because SwiftPM's manifest tooling fails
     with some Command Line Tools versions; `build-app.sh` picks the newest SDK the compiler
     accepts. Keep it that way.
   - `security add-trusted-cert` may pop a password dialog — that is macOS, expected.
3. **Verify the server without the app:** `server/.venv/bin/python -m pytest -q server/tests`
   must pass. Then `scripts/run.sh` in the background, wait for `curl -s
   http://127.0.0.1:8765/health` → `ready`, run `server/.venv/bin/python
   server/bench/make_samples.py` (needs `say` + ffmpeg) and
   `server/.venv/bin/python server/bench/stream_client.py`. Expect 14 captions, three speakers
   (S1/S2/S3), full-caption p50 around 2.5–3 s on an M-series Mac. Stop the server afterwards
   (`lsof -ti tcp:8765 | xargs kill`) so the app can manage its own.
4. **App:** `/Applications/LiveMandarin.app` should be running (menu-bar speech bubble; no
   Dock icon — that is by design, `LSUIElement`). Ask the user to press ▶ in the panel and
   allow the **System Audio Recording** prompt. If they see *"The user declined TCCs"* or no
   prompt: `tccutil reset ScreenCapture com.sanket.livemandarin && tccutil reset AudioCapture
   com.sanket.livemandarin`, relaunch the app, press ▶ again. Confirm success from
   `~/Library/Logs/LiveMandarin-app.log` ("audio capture started", "server link: connected")
   and `~/Library/Logs/LiveMandarin.log` ("client connected", "audio: … frames received").
5. **Tailor:** ask for a handful of names/products/jargon from their meetings and put them in
   `server/glossary.yaml` (`asr_hints` ≤ 30 words; `terms` for fixed translations). Set
   `caption.mode` if they mainly need English → Chinese or both directions.

You cannot see their screen: for anything visual (panel visible? captions appearing?) ask
them, and read the two log files for the facts.

## Working in this codebase

- **Measure before changing models or thresholds.** Every model choice here came from
  `server/bench/`; keep it that way. Re-run `bench_asr.py` / `bench_mt.py` / `stream_client.py`
  and quote numbers when proposing a change. Cloud translation was considered and deliberately
  not added (the owner wants zero cost and privacy); do not add cloud calls unless asked.
- **Protocol** between app and server is documented at the top of `server/app.py`
  (`listening / processing / partial / final / drop`, `config` for the direction). Change both
  sides together; `bench/stream_client.py` is the reference client.
- **Tests:** `server/.venv/bin/python -m pytest -q server/tests` — no models needed; keep them
  fast. Add a test when you touch segmentation, prompt building, speaker clustering, language
  routing, or the YAML files (`test_config.py` catches malformed config edits, which otherwise
  stop the server without a visible error).
- **Config edits:** `config.yaml` and `glossary.yaml` are read by both humans and code. Edit
  them as YAML (or rewrite the whole file), never with substring replacement — `speakers:` is
  a prefix of `max_speakers:`.
- **Swift build:** `scripts/build-app.sh` compiles, signs, installs to `/Applications` and
  kills the running instance; relaunch with `open "/Applications/LiveMandarin.app"`. Grep the
  build output for `error:` — warnings about `Sendable` are expected. The app must stay
  signed with the "LiveMandarin Local" identity or the user re-grants permissions each build.
- **Logging:** the server logs timings only (`server.log_text: false`). Do not log caption
  text by default; it is meeting content.
- **Known non-obvious facts** (see code comments): Hy-MT2 1.8B vs 7B chat formats differ and
  Ollama's auto template for the 1.8B is broken (`mt.py` renders raw prompts); the recognizer
  runs in a child process for speed and keeps the GPU/core warm only while someone speaks; a
  long `asr_hints` list hurts accuracy; streaming and speculative modes of `mlx-qwen3-asr`
  were tested and rejected.
- **Style:** small files, comments explain *why*; Python matches the existing type-hinted,
  asyncio style; Swift is plain SwiftUI/AppKit with no dependencies. Keep the UI minimal.
