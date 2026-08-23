"""Prepare arabic-sadeed-imf/test.jsonl on rababa-datasets for ara-diac parity.

Reads sadeed-hf/test.txt (single-column diacritized lines), derives
undiacritized src, writes {src, tgt} jsonl next to the other IMF test
sets. One-shot; idempotent via DONE marker.

Usage:
    modal run prep_arabic_imf_test.py
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
N_TEST = 4000

image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("prep-arabic-imf-test", image=image)


@app.function(timeout=10 * 60, volumes={"/datasets": datasets_volume})
def prep() -> dict:
    import json

    datasets_volume.reload()
    out_dir = Path("/datasets/arabic-sadeed-imf")
    marker = out_dir / "DONE"
    if marker.exists():
        return {"status": "already-done"}
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path("/datasets/sadeed-hf/test.txt")
    rows = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        src = DIACRITICS_RE.sub("", line).strip()
        if src and 2 <= len(src) <= 1200:
            rows.append({"src": src, "tgt": line})
        if len(rows) >= N_TEST:
            break
    (out_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    marker.touch()
    datasets_volume.commit()
    return {"rows": len(rows)}


@app.local_entrypoint()
def main():
    print(prep.remote())
