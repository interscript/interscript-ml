#!/usr/bin/env bash
# Export a trained task to ONNX. Usage: scripts/export.sh <task_name>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "Usage: $0 <task_name>" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

python -m src.cli export --task "$TASK" --data-root data --out-root "models/$TASK"
