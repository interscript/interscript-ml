#!/usr/bin/env bash
# Fetch raw datasets. Does NOT clean or encode — that's the data
# module's job. Place files under data/raw/<source>.{txt,tsv}.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p data/raw

# Tashkeela++ (rababa_arabic)
# Source: https://huggingface.co/datasets/community/tashkeela_plus_plus
TASHKEELA="${TASHKEELA:-}"
if [[ -n "$TASHKEELA" ]]; then
  echo "Fetching Tashkeela++ from $TASHKEELA"
  curl -sSL "$TASHKEELA" -o data/raw/tashkeela_plus_plus.txt
fi

# Wiktionary Thai-IPA (secryst_thai_ipa)
# Source: scraped from en.wiktionary.org Appendix:Thai_pronunciation
WIKTIONARY="${WIKTIONARY:-}"
if [[ -n "$WIKTIONARY" ]]; then
  echo "Fetching Wiktionary Thai-IPA from $WIKTIONARY"
  curl -sSL "$WIKTIONARY" -o data/raw/wiktionary_thai_ipa.tsv
fi

# SNA Nikud (rababa_hebrew)
# Source: https://github.com/.dicta-center/SNA
SNA_NIKUD="${SNA_NIKUD:-}"
if [[ -n "$SNA_NIKUD" ]]; then
  echo "Fetching SNA Nikud from $SNA_NIKUD"
  curl -sSL "$SNA_NIKUD" -o data/raw/sna_nikud.txt
fi

echo "Done. Files in data/raw/"
ls -lh data/raw/ || true
