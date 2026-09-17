# Meet Captions

Live translated captions for calls, fully local on a Mac. Built for understanding Mandarin
speakers in Google Meet; works with any app that plays audio (Meet, Zoom, Teams, FaceTime, a
video in the browser) and, optionally, the microphone for in-person conversations.

- **Chinese → English, English → Chinese, or both** — chosen per sentence by the detected language
- **Speaker colors** — each voice gets its own color for the whole call
- **Streams as it translates** — first English words ≈1.8 s after the speaker pauses
- **Private and free** — no cloud, no accounts; nothing leaves the machine
- **Native Mac app** — menu-bar icon, floating panel that stays on top of any app or Space

```
any app's audio ──► Meet Captions.app (ScreenCaptureKit) ──► local Python server ──► floating panel
                                                              │  Silero VAD      finds sentence pauses
                                                              │  Qwen3-ASR 1.7B  speech → text + language detection
                                                              │  CAM++           voice print → speaker color
                                                              └  Hy-MT2 1.8B     translation, streamed word by word
```

## Requirements

- Apple Silicon Mac (M1 or newer), macOS 14 or later, ~6 GB free disk, 16 GB RAM recommended
  (the pipeline uses ≈4.5 GB while running)
- [Homebrew](https://brew.sh) and Apple's Command Line Tools (`xcode-select --install`)

## Install

```bash
git clone https://github.com/shahsanket2107/meet-captions.git && cd meet-captions
scripts/setup.sh
```

The script installs Ollama and Python, downloads the three models (~5 GB), runs the tests,
creates a local signing certificate, builds the app, installs it to `/Applications`, and opens
it. It is safe to re-run. **Using Claude Code?** Open the repo and say "set this up" —
`CLAUDE.md` tells it exactly what to do and how to verify it.

First use: press **▶** in the caption panel. macOS asks once for **System Audio Recording**
permission — allow it (and relaunch the app if macOS asks). If you enable the microphone, it
asks for that once too.

## Use

| Where | What |
|---|---|
| **Menu-bar icon** (speech bubble) | Start/Stop · Show panel · Clear · **Translate** direction · **Microphone** on/off · Launch at login · Quit |
| **Panel header** | ● status + live audio level · direction badge · A− / A+ text size · **Orig** (show the original sentence) · ⤓ save transcript · 🗑 clear · ▶/■ · ✕ hide |
| **Panel body** | One line per sentence; colored stripe = speaker; gray text = translation in progress; typing dots = someone is speaking |

- The panel floats above everything, including full-screen calls. Drag it by the header,
  resize from any edge; position and size are remembered.
- **Direction:** *Chinese → English* (default) captions only Chinese speech (Mandarin,
  Cantonese, dialects, mixed sentences); *English → Chinese* the reverse; *Both* translates each
  sentence by its detected language. Speech in other languages is ignored.
- **Microphone:** off by default. Turn it on to caption your own voice or people in the room
  (uses the input device selected in System Settings → Sound). With a call playing on
  loudspeakers the mic re-hears the call and captions duplicate — use headphones then.
- The app starts the caption server itself when you press ▶ ("Loading models…" for ~10 s) and
  stops it when you quit. Nothing runs when nobody is speaking.

### Make it accurate for your meetings

Edit `server/glossary.yaml` at any time (applies to the next sentence, no restart):

- `asr_hints` — English words people say mid-sentence: product names, teammates, jargon.
  Keep it to **15–30 words**; a short relevant list halved recognition errors in testing, a
  96-word list measured as bad as none.
- `terms` — fixed translations (`回滚 ↔ rollback`), applied in both directions.

### Settings (`server/config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `caption.mode` | `zh-en` | Default direction (`zh-en`, `en-zh`, `auto`); the menu overrides it live |
| `segmenter.min_silence_ms` | 400 | Pause that ends a sentence — lower = faster captions, more fragments |
| `segmenter.soft_max_s` / `hard_max_s` | 6 / 10 | Long monologues are cut at a short pause after 6 s, unconditionally at 10 s |
| `mt.history_size` | 3 | Previous sentences given to the translator as context |
| `speakers.threshold` | 0.7 | Voice similarity to count as the same person (raise if two people merge, lower if one person splits) |
| `speakers.max_speakers` | 8 | Distinct voices tracked per session |
| `server.log_text` | false | Write caption text to the log (off: timings only) |

## Models

All models run on-device and are downloaded once by `scripts/setup.sh`.

| Stage | Model | Params | Disk | License | Role |
|---|---|---|---|---|---|
| Voice activity | [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX) | ~1 M | 2 MB | MIT | Decides when someone is speaking and where sentences end, so the large models only run on speech |
| Speech → text | [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR), 8-bit, via [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr) | 1.7 B | 2.3 GB | Apache-2.0 | Transcribes 30 languages + 22 Chinese dialects, handles Chinese/English code-switching, reports the language (used to pick the direction), takes glossary words as spelling hints |
| Speaker ID | [CAM++ zh-en](https://github.com/k2-fsa/sherpa-onnx) (3D-Speaker, sherpa-onnx) | ~7 M | 28 MB | Apache-2.0 | 192-d voice embedding per sentence; online clustering keeps the same color per person (~40 ms on CPU, runs alongside recognition) |
| Translation | [Hy-MT2-1.8B](https://github.com/Tencent-Hunyuan/Hy-MT2), Q8 GGUF, via [Ollama](https://ollama.com) | 1.8 B | 1.9 GB | Tencent Hunyuan license (free) | Translation-only model, 33 languages, Chinese↔English core pair; follows terminology instructions; previous sentences are passed as prior chat turns for context |

Why these (all measured, see `server/bench/`):

- **Qwen3-ASR** is the strongest open Mandarin recognizer available; on the project's technical
  test set it scored a **2.9% character error rate** with a short glossary (6.5% without). The
  0.6B and 4-bit variants were 2–5× worse; Whisper-large is ~2× worse on Chinese in published
  numbers. 8-bit weights gave identical accuracy to full precision at 30% less time.
- **Hy-MT2-1.8B** beats far larger general models on Chinese↔English in Tencent's benchmarks
  and translates a sentence in ~0.85 s here. Its 7B sibling was slightly better on nuance but
  2–3× slower; 4-bit was only 20% faster and introduced grammar slips.
- Two specialist ~2B models beat one large general model on speed, memory (4.5 GB total) and
  accuracy for this job. Cloud models would translate more fluently but cost money and send
  meeting content off the machine; the local pipeline was chosen deliberately.

Measured on an M4 / 16 GB: first translated words **≈1.8 s** after the speaker pauses, full
sentence **≈2.4 s** (p95 ≈2.9 s); speaker labels 14/14 correct across three voices.

## Privacy

Audio is captured from the Mac's own output (and the mic only if you enable it), processed
locally, and discarded. Captions stay in the panel until you clear them or quit; a transcript is
written only when you press ⤓. The server log holds timings only unless `log_text` is on.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "The user declined TCCs…" or no permission prompt | macOS remembers a stale grant. `tccutil reset ScreenCapture com.sanket.meetcaptions`, relaunch, press ▶ again. If it recurs after rebuilds, the signing certificate is missing: `scripts/make-signing-cert.sh && scripts/build-app.sh` |
| "Live" but level bars never move | The app isn't receiving audio: check System Settings → Privacy & Security → Screen & System Audio Recording. The app restarts a stalled capture automatically |
| "Caption server didn't become ready" | Look at `~/Library/Logs/MeetCaptions.log`. Usually Ollama isn't running (`brew services start ollama`) or a model is missing (re-run `scripts/setup.sh`) |
| Build errors about SDK / compiler mismatch | Command Line Tools are stale: `sudo mv /Library/Developer/CommandLineTools{,.old}; xcode-select --install`. `build-app.sh` uses plain `swiftc` and picks the newest SDK the compiler accepts, so SwiftPM is not required |
| Two people share a color / one person gets two | Adjust `speakers.threshold` (default 0.7): raise to split, lower to merge |
| A technical term keeps coming out wrong | Add it to `asr_hints` (recognition) or `terms` (translation) in `glossary.yaml` |

Logs: `~/Library/Logs/MeetCaptions.log` (server), `~/Library/Logs/MeetCaptions-app.log` (app).

## Project layout

```
app/Sources/MeetCaptions/   Swift menu-bar app (SwiftUI + ScreenCaptureKit + AVFoundation)
  MeetCaptionsApp.swift     menu, coordinator (server lifecycle, capture, watchdog)
  AudioCapture.swift        system audio → 16 kHz PCM frames    MicCapture.swift  optional mic
  CaptionClient.swift       WebSocket to the server             CaptionPanel.swift floating panel UI
  CaptionModel.swift        observable state
app/Info.plist, app/AppIcon.icns
server/
  app.py                    WebSocket server; protocol documented at the top of the file
  segmenter.py              Silero VAD + sentence cutting     asr.py       Qwen3-ASR in a child process
  mt.py                     Hy-MT2 prompts (official templates), streaming Ollama client
  speakers.py               voice embeddings + online clustering
  config.yaml, glossary.yaml, requirements.txt
  bench/                    make_samples.py (synthetic test meeting), bench_asr.py, bench_mt.py,
                            stream_client.py (end-to-end latency), samples/script.jsonl
  tests/                    pytest: segmenter, prompts, speakers, language routing, config files
scripts/                    setup.sh · run.sh (server only) · build-app.sh · make-signing-cert.sh · make-icon.swift
```

## Development

```bash
cd server
.venv/bin/python -m pytest -q tests           # unit tests (no models needed)
.venv/bin/python bench/make_samples.py        # synthesize the 14-sentence test meeting (macOS voices + ffmpeg)
scripts/run.sh                                # run the server in a terminal
.venv/bin/python bench/stream_client.py       # stream the test meeting through it; prints latencies + speakers
.venv/bin/python bench/bench_asr.py           # recognizer accuracy/speed table
.venv/bin/python bench/bench_mt.py            # translator quality/speed
scripts/build-app.sh                          # rebuild + reinstall the app (keeps the permission)
```

Design notes that are easy to get wrong (details in the code comments):

- Hy-MT2's 1.8B and 7B GGUFs use different chat formats and Ollama's auto-converted template
  for the 1.8B is broken — `mt.py` renders raw prompts and detects the format from the BOS token.
- Context is passed as previous chat turns, not Hy-MT2's "background information" template,
  which made the model re-translate the context.
- The recognizer runs in a separate process: in-process it measured 1.1–1.9 s per sentence
  versus 0.85 s alone (thread scheduling / interpreter lock effects). While someone speaks, the
  process keeps the GPU and its core lightly busy; otherwise the next transcription is ~0.6 s slower.
- The app is built with plain `swiftc` (`scripts/build-app.sh`), not SwiftPM, and signed with a
  local self-signed certificate so macOS keeps the capture permission across rebuilds.
