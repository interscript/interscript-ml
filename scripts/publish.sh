#!/usr/bin/env bash
# Publish a trained + exported model to HuggingFace Hub.
# Requires HF_TOKEN env var.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "Usage: $0 <task_name>" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN env var not set" >&2
  exit 2
fi

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

python -m src.cli publish --task "$TASK" --out-root "models/$TASK" --repo "interscript/$TASK"
