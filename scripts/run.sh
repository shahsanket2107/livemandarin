#!/usr/bin/env bash
# Start Ollama (if needed) and the caption server. Ctrl-C stops the server; Ollama keeps running.
set -euo pipefail
cd "$(dirname "$0")/../server"

if ! pgrep -x ollama >/dev/null; then
  echo "starting ollama…"
  (OLLAMA_FLASH_ATTENTION=1 ollama serve >/tmp/ollama.log 2>&1 &)
  for _ in $(seq 1 20); do
    curl -sf http://127.0.0.1:11434/ >/dev/null && break
    sleep 0.5
  done
fi

exec .venv/bin/python app.py
