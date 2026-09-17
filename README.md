# LiveMandarin — real-time Mandarin ↔ English translation captions for any call on your Mac

**Free, offline, on-device.** LiveMandarin listens to whatever your Mac is playing — a Google
Meet, Zoom or Teams call, FaceTime, a video, a podcast — or to the microphone for in-person
conversations, and shows live translated captions in a floating panel. Chinese speech becomes
English, English becomes Chinese, each sentence in about two seconds, with a color per speaker.
Nothing leaves your computer.

> Understand Mandarin-speaking colleagues in meetings, follow Chinese videos and livestreams,
> practice the language with instant subtitles, or let a Chinese speaker read your English —
> without a subscription, an API key, or an internet connection.

<!-- Add a screenshot here: docs/screenshot.png -->

## Why LiveMandarin

| | LiveMandarin | Google Meet captions | Cloud translation extensions |
|---|---|---|---|
| Cost | **Free, unlimited** | Needs a paid Workspace plan for translation | Subscription / minute caps |
| Where it runs | **On your Mac** | Google's servers | Vendor's servers |
| Works with | **Any app** (Meet, Zoom, Teams, FaceTime, YouTube, Bilibili, the room) | Meet only | Usually one app / browser tab |
| Mandarin accuracy | **Qwen3-ASR, the strongest open Chinese recognizer**, with your own glossary | Good | Varies |
| Direction | Chinese → English, English → Chinese, or both automatically | Limited pairs | Varies |
| Speakers | Color per voice | Names (Meet only) | Rarely |
| Privacy | Audio never leaves the machine | Sent to Google | Sent to the vendor |

## How it works

```
any app's audio ──► LiveMandarin.app (ScreenCaptureKit) ──► local Python server ──► floating caption panel
                                                             │  Silero VAD       finds sentence pauses
                                                             │  Qwen3-ASR 1.7B   speech → text + language detection
                                                             │  CAM++            voice print → speaker color
                                                             └  Hy-MT2 1.8B      translation, streamed word by word
```

Measured on a MacBook Pro M4 / 16 GB with a technical test meeting: **2.9% character error
rate** on mixed Mandarin/English speech, first translated words **≈1.8 s** after the speaker
pauses, full sentence **≈2.4 s**, about **4.5 GB of memory** while running and nothing when
nobody speaks. Details and benchmarks in [Models](#models) and `server/bench/`.

## Install (one command)

Requirements: an Apple Silicon Mac (M1 or newer), macOS 14+, ~6 GB free disk,
[Homebrew](https://brew.sh). The setup script installs everything else.

```bash
git clone https://github.com/shahsanket2107/livemandarin.git && cd livemandarin
scripts/setup.sh
```

That installs Ollama, Python and ffmpeg via Homebrew, downloads the three AI models (~5 GB),
runs the test suite, creates a local signing certificate, builds the native app, installs it
to `/Applications` and opens it. Safe to re-run at any time.

**Using Claude Code?** Open the folder and say *"set this up"* — [`CLAUDE.md`](CLAUDE.md) walks
it through installing, verifying and tailoring everything for you.

First use: press **▶** in the caption panel. macOS asks once for **System Audio Recording**
permission — allow it (relaunch the app if macOS asks). If you enable the microphone, it asks
for that once too.

## Use

| Where | What |
|---|---|
| **Menu-bar icon** (speech bubble) | Start/Stop · Show panel · Clear · **Translate** direction · **Microphone** on/off · Launch at login · Quit |
| **Panel header** | ● status + live audio level · direction badge · A− / A+ text size · **Orig** (show the original sentence) · ⤓ save transcript · 🗑 clear · ▶/■ · ✕ hide |
| **Panel body** | One line per sentence; colored stripe = speaker; gray text = translation in progress; typing dots = someone is speaking |

- The panel floats above everything, including full-screen calls. Drag it by the header,
  resize from any edge; position and size are remembered.
- **Direction:** *Chinese → English* (default) captions only Chinese speech — Mandarin,
  Cantonese, regional dialects, and mixed Chinese/English sentences; *English → Chinese* the
  reverse; *Both* translates each sentence by its detected language. Other languages are ignored.
- **Microphone:** off by default. Turn it on to caption your own voice or the people in the
  room (uses the input device selected in System Settings → Sound). If a call is playing on
  loudspeakers the mic re-hears it and captions duplicate — use headphones then.
- The app starts the caption server itself when you press ▶ ("Loading models…" for ~10 s) and
  stops it when you quit.

### Make it accurate for your meetings

Edit `server/glossary.yaml` at any time (applies to the next sentence, no restart):

- `asr_hints` — English words people say mid-sentence: product names, teammates, jargon.
  Keep it to **15–30 words**; a short relevant list halved recognition errors in testing,
  while a 96-word list measured as bad as none.
- `terms` — fixed translations (`回滚 ↔ rollback`), applied in both directions.

### Settings (`server/config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `caption.mode` | `zh-en` | Default direction (`zh-en`, `en-zh`, `auto`); the menu overrides it live |
| `segmenter.min_silence_ms` | 400 | Pause that ends a sentence — lower = faster captions, more fragments |
| `segmenter.soft_max_s` / `hard_max_s` | 6 / 10 | Long monologues are cut at a short pause after 6 s, unconditionally at 10 s |
| `mt.history_size` | 3 | Previous sentences given to the translator as context |
| `speakers.threshold` | 0.7 | Voice similarity to count as the same person (raise if two people merge, lower if one splits) |
| `speakers.max_speakers` | 8 | Distinct voices tracked per session |
| `server.log_text` | false | Write caption text to the log (off: timings only) |

## Models

All models run on-device and are downloaded once by `scripts/setup.sh`.

| Stage | Model | Params | Disk | License | Role |
|---|---|---|---|---|---|
| Voice activity | [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX) | ~1 M | 2 MB | MIT | Decides when someone is speaking and where sentences end, so the large models only run on speech |
| Speech → text | [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR), 8-bit, via [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr) | 1.7 B | 2.3 GB | Apache-2.0 | Transcribes 30 languages + 22 Chinese dialects, handles Chinese/English code-switching, reports the language (used to pick the direction), takes glossary words as spelling hints |
| Speaker ID | [CAM++ zh-en](https://github.com/k2-fsa/sherpa-onnx) (3D-Speaker, sherpa-onnx) | ~7 M | 28 MB | Apache-2.0 | 192-d voice embedding per sentence; online clustering keeps the same color per person (~40 ms on CPU, alongside recognition) |
| Translation | [Hy-MT2-1.8B](https://github.com/Tencent-Hunyuan/Hy-MT2), Q8 GGUF, via [Ollama](https://ollama.com) | 1.8 B | 1.9 GB | Tencent Hunyuan license (free) | Translation-only model, 33 languages, Chinese↔English core pair; follows terminology instructions; previous sentences are passed as prior chat turns for context |

Why these — every choice was measured (`server/bench/`):

- **Qwen3-ASR** is the strongest open Mandarin speech recognizer available: 2.9% character
  error rate on the project's technical test set with a short glossary (6.5% without). The
  0.6B and 4-bit variants were 2–5× worse; Whisper-large is roughly 2× worse on Chinese in
  published numbers. 8-bit weights matched full precision at 30% less time.
- **Hy-MT2-1.8B** beats far larger general-purpose models on Chinese↔English in Tencent's
  benchmarks and translates a sentence in ~0.85 s here. Its 7B sibling was slightly better on
  nuance but 2–3× slower; a 4-bit build was only 20% faster and introduced grammar slips.
- Two specialist ~2B models beat one large general model on speed, memory and accuracy for
  this job. Cloud models translate more fluently but cost money and send your meetings off
  the machine; the fully local pipeline is a deliberate choice.

## Privacy

Audio is captured from the Mac's own output (and the mic only if you enable it), processed
locally, and discarded. Captions stay in the panel until you clear them or quit; a transcript
is written only when you press ⤓. The server log holds timings only unless `log_text` is on.

## FAQ

**Does it work with Zoom / Teams / FaceTime / YouTube / Bilibili?** Yes — anything that plays
audio on the Mac. It captures the system audio stream, not a specific app.

**Does it need the internet?** Only for the one-time model download. Captioning is fully offline.

**Can it caption my own voice, or a conversation in the room?** Yes — turn on the microphone
in the menu. Use headphones if a call is also playing.

**Does it show people's names?** It shows a consistent color per voice. Call audio carries no
identities, so names aren't possible offline.

**Cantonese? Taiwanese Mandarin? Mixed Chinese and English?** All recognized; sentences that
mix English terms into Chinese are handled and the English terms are kept.

**Intel Mac / Windows / Linux?** Not currently: the speech model runs on Apple's MLX framework
and audio capture uses ScreenCaptureKit. The Python server is portable; the app is not.

**Other language pairs?** The recognizer supports 30 languages and the translator 33, so it's
a small change in `server/app.py` (`direction_for`) and `mt.py` (prompt templates). Mandarin
↔ English is what has been tuned and tested.

**How much does it cost to run?** Nothing. About 4.5 GB of memory while captions are on.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "The user declined TCCs…" or no permission prompt | macOS remembers a stale grant. `tccutil reset ScreenCapture com.sanket.livemandarin`, relaunch, press ▶ again. If it recurs after rebuilds, the signing certificate is missing: `scripts/make-signing-cert.sh && scripts/build-app.sh` |
| "Live" but the level bars never move | The app isn't receiving audio: check System Settings → Privacy & Security → Screen & System Audio Recording. The app restarts a stalled capture automatically |
| "Caption server didn't become ready" | Look at `~/Library/Logs/LiveMandarin.log`. Usually Ollama isn't running (`brew services start ollama`) or a model is missing (re-run `scripts/setup.sh`) |
| Build errors about SDK / compiler mismatch | Command Line Tools are stale: `sudo mv /Library/Developer/CommandLineTools{,.old}; xcode-select --install`. `build-app.sh` uses plain `swiftc` and picks the newest SDK the compiler accepts, so SwiftPM is not required |
| Two people share a color / one person gets two | Adjust `speakers.threshold` (default 0.7): raise to split, lower to merge |
| A technical term keeps coming out wrong | Add it to `asr_hints` (recognition) or `terms` (translation) in `glossary.yaml` |

Logs: `~/Library/Logs/LiveMandarin.log` (server), `~/Library/Logs/LiveMandarin-app.log` (app).

## Project layout

```
app/Sources/LiveMandarin/   Swift menu-bar app (SwiftUI + ScreenCaptureKit + AVFoundation)
  LiveMandarinApp.swift     menu, coordinator (server lifecycle, capture, watchdog)
  AudioCapture.swift        system audio → 16 kHz PCM frames    MicCapture.swift   optional mic
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
  versus 0.85 s alone. While someone speaks, the process keeps the GPU and its core lightly
  busy; otherwise the next transcription is ~0.6 s slower.
- The app is built with plain `swiftc` (`scripts/build-app.sh`), not SwiftPM, and signed with a
  local self-signed certificate so macOS keeps the capture permission across rebuilds.

## Contributing

Issues and pull requests are welcome — especially real-call accuracy reports (which words the
recognizer or translator got wrong), other language pairs, and Intel/Windows/Linux ports of the
server. Please run the tests and, for model or threshold changes, the benchmarks, and include
the numbers.

## License

MIT for this repository's code. The models keep their own licenses (see the table above).

---

*Keywords: Mandarin to English live translation, Chinese to English real-time subtitles,
Google Meet translation captions, Zoom live translation, offline speech translation macOS,
on-device AI interpreter, Qwen3-ASR, Hy-MT2, Ollama, Apple Silicon, MLX.*
