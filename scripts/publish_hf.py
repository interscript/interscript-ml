"""Publish a models.yaml entry to the Hugging Face org mirror.

One command, both channels: the GitHub Releases asset stays the
runtime channel; this adds the HF mirror (riboseinc/<id>) with a
generated model card.

Usage:
    python3 scripts/publish_hf.py --id ara-diac-2.0 --zip /path/to/ara-diac-2.0-fp32.zip --dry-run
    python3 scripts/publish_hf.py --id ara-diac-2.0 --zip /path/...   # uploads; needs HF_TOKEN

--zip accepts the assembled zip (parts reassembled: cat *.part-* > zip).
Variant ids (e.g. ara-diac-2.0-int8) are separate repos: run once per id.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_card import hf_repo_for, render_card  # noqa: E402

INDEX = "models.yaml"
CARDS_DIR = Path(__file__).resolve().parent.parent / "cards"


def load_entry(model_id: str) -> dict:
    index = yaml.safe_load(Path(INDEX).read_text(encoding="utf-8"))["models"]
    entry = index.get(model_id)
    if not entry:
        raise SystemExit(f"{model_id} not in {INDEX}")
    return entry


def variants_of(model_id: str, index: dict) -> list[str]:
    return [k for k in index if k != model_id and k.startswith(model_id + "-")]


def curated_body(model_id: str) -> str:
    card = CARDS_DIR / f"{model_id}.md"
    return card.read_text(encoding="utf-8") if card.exists() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = yaml.safe_load(Path(INDEX).read_text(encoding="utf-8"))["models"]
    entry = load_entry(args.id)
    repo = hf_repo_for(args.id)
    card = render_card(args.id, entry, variants_of(args.id, index),
                       body=curated_body(args.id))
    zipped = args.zip.read_bytes()

    print(f"repo:      {repo}")
    print(f"artifact:  {args.zip} ({len(zipped)} bytes; index sha256 "
          f"{entry.get('sha256', '?')[:16]}…)")
    print(f"card:      {len(card)} chars, {len(card.splitlines())} lines")
    if args.dry_run:
        print(card)
        print("[dry-run] no upload performed")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (fine-grained, write-scoped to the repo)")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode("utf-8"),
                    path_in_repo="README.md", repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(args.zip),
                    path_in_repo=entry.get("filename", args.zip.name),
                    repo_id=repo, repo_type="model")
    classes = args.zip.with_name("classes.json")
    if classes.exists():  # plane artifacts carry the class inventory
        api.upload_file(path_or_fileobj=str(classes),
                        path_in_repo="classes.json", repo_id=repo, repo_type="model")
    print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
