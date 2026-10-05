"""Assemble a model artifact from its index parts, sha-verified.

Downloads every part URL in the entry (in order), verifies each part's
sha256 against the entry, concatenates, verifies the whole-file sha256,
and writes the artifact. Used by the hf-publish workflow and anywhere a
release asset must be reassembled without trusting the transport.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, expected_sha: str, label: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as r:
        data = r.read()
    got = _sha(data)
    if got != expected_sha:
        raise ValueError(f"{label}: sha256 mismatch (got {got[:16]}…, want {expected_sha[:16]}…)")
    return data


def assemble(entry: dict, out: Path, index_dir: Path | None = None) -> Path:
    parts = entry.get("parts") or []
    chunks: list[bytes] = []
    if not parts and entry.get("url"):
        chunks = [_download(entry["url"], entry.get("sha256", ""), entry["url"].rsplit("/", 1)[-1])]
    for p in parts:
        name = p["url"].rsplit("/", 1)[-1]
        chunks.append(_download(p["url"], p["sha256"], name))
    whole = b"".join(chunks)
    want = entry.get("sha256")
    if want and _sha(whole) != want:
        raise ValueError(f"assembled artifact sha256 mismatch (want {want[:16]}…)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(whole)
    return out
