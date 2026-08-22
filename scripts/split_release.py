#!/usr/bin/env python3
"""Split a release asset to fit GitHub's 2 GiB per-asset cap.

Writes ``<zip>.part-00``, ``<zip>.part-01``, ... alongside the source,
prints per-part sha256s, and emits a ``parts:`` block for models.yaml.
Runtimes reassemble by plain byte concatenation; the whole-file sha256
remains the index contract, each part carries its own sha256 so a
corrupt part is identified, not just "the download failed".

    python scripts/split_release.py models/heb-diac/heb-diac-1.0-fp32.zip \
        --url-base https://github.com/interscript/interscript-ml/releases/download/heb-diac-1.0
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# GitHub hard-caps release assets at 2,147,483,648 bytes; stay well clear.
DEFAULT_PART_SIZE = 1_500_000_000
CHUNK = 1024 * 1024


def split(src: Path, part_size: int) -> list[tuple[Path, str, int]]:
    parts: list[tuple[Path, str, int]] = []
    with src.open("rb") as fh:
        index = 0
        while True:
            remaining = part_size
            digest = hashlib.sha256()
            part_path = src.parent / f"{src.name}.part-{index:02d}"
            written = 0
            with part_path.open("wb") as out:
                while remaining > 0:
                    chunk = fh.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                    written += len(chunk)
            if written == 0:
                part_path.unlink()
                break
            parts.append((part_path, digest.hexdigest(), written))
            print(f"{part_path.name}  {written:>13,} bytes  {digest.hexdigest()}")
            index += 1
            if written < part_size:
                break
    return parts


def yaml_block(parts: list[tuple[Path, str, int]], url_base: str) -> str:
    lines = ["    parts:"]
    for part_path, sha256, size in parts:
        url = f"{url_base.rstrip('/')}/{part_path.name}" if url_base else part_path.name
        lines.append(f"      - url: {url}")
        lines.append(f"        sha256: {sha256}")
        lines.append(f"        size: {size}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--url-base", default="")
    args = parser.parse_args()

    src = args.zip.resolve()
    parts = split(src, args.part_size)
    total = hashlib.sha256()
    with src.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            total.update(chunk)
    print(f"\nwhole-file sha256: {total.hexdigest()}  ({src.stat().st_size:,} bytes)")
    print("\nmodels.yaml entry:")
    print(yaml_block(parts, args.url_base))


if __name__ == "__main__":
    main()
