#!/usr/bin/env bash
# One-time setup on a Mac: Python env, dependencies, model downloads (~5 GB), signing
# certificate, and the native app. Safe to re-run; each step is skipped if already done.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Checking prerequisites"
if [ "$(uname -m)" != "arm64" ]; then
  echo "Apple Silicon (M1 or newer) is required: the speech model runs on MLX." >&2; exit 1
fi
if ! command -v brew >/dev/null; then
  echo "Homebrew is required: https://brew.sh" >&2; exit 1
fi
if ! xcode-select -p >/dev/null 2>&1 || ! swiftc -version >/dev/null 2>&1; then
  xcode-select --install 2>/dev/null || true
  echo "Apple's Command Line Tools are being installed: click Install in the dialog, then re-run this script." >&2
  exit 1
fi
brew list --versions ollama >/dev/null 2>&1 || brew install ollama
brew list --versions python@3.13 >/dev/null 2>&1 || brew install python@3.13
brew list --versions ffmpeg >/dev/null 2>&1 || brew install ffmpeg   # only for regenerating the benchmark audio

echo "==> Python environment"
cd "$ROOT/server"
if [ ! -x .venv/bin/python ]; then
  "$(brew --prefix)/bin/python3.13" -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "==> Translation model (Ollama)"
if ! pgrep -x ollama >/dev/null; then
  (ollama serve >/tmp/ollama.log 2>&1 &)
  sleep 2
fi
MT_MODEL=$(.venv/bin/python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['mt']['model'])")
ollama pull "$MT_MODEL"

echo "==> Speech model (Hugging Face cache)"
ASR_MODEL=$(.venv/bin/python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['asr']['model'])")
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$ASR_MODEL')"

echo "==> Speaker-label model"
SPK_MODEL=$(.venv/bin/python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['speakers']['model'])")
[ -f "$SPK_MODEL" ] || curl -L -o "$SPK_MODEL" --create-dirs \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx

echo "==> Tests"
.venv/bin/python -m pytest -q tests

echo "==> Signing certificate and app"
"$ROOT/scripts/make-signing-cert.sh"
"$ROOT/scripts/build-app.sh"

echo
echo "Setup complete. LiveMandarin is installed in /Applications and running in the menu bar."
echo "Press ▶ in its panel; allow the system-audio permission when macOS asks."
open "/Applications/LiveMandarin.app"
