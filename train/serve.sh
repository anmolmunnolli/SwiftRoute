#!/usr/bin/env bash
# Serve a model (optionally with the LoRA adapter) on an OpenAI-compatible endpoint with
# tool calling. The agent talks to it exactly as it talks to Ollama:
#   SWIFTROUTE_LLM_URL=http://localhost:8080/v1 SWIFTROUTE_MODEL=default_model python -m swiftroute.voice.bot
#
#   bash train/serve.sh                  # 1.5B + SwiftRoute adapter (the student)
#   ADAPTER= bash train/serve.sh         # 1.5B base, no adapter (the "before" for the comparison)
set -euo pipefail
cd "$(dirname "$0")/.."
STUDENT="${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER="${ADAPTER-data/adapters/$(basename "$STUDENT")-swiftroute}"
PORT="${PORT:-8080}"

args=(--model "$STUDENT" --port "$PORT" --temp 0.2 --max-tokens 160)
if [ -n "$ADAPTER" ]; then args+=(--adapter-path "$ADAPTER"); fi
echo "serving $STUDENT ${ADAPTER:+with $ADAPTER} on http://localhost:$PORT/v1 (request model: default_model)"
exec mlx_lm.server "${args[@]}"
