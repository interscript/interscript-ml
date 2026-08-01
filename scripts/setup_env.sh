#!/usr/bin/env bash
# Create venv + install deps. Idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
pip install -e ".[train]" || echo "WARN: train extras skipped (no torch in this env)"
pip install -e ".[export]" || echo "WARN: export extras skipped (no onnxruntime)"

echo "Environment ready. Activate with: source .venv/bin/activate"
