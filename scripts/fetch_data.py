"""Fetch raw corpora from HuggingFace Hub.

Replaces ``scripts/fetch_data.sh``. The shell version required manual
env-var URLs; this one knows the canonical dataset locations and
validates downloads.

For ``rababa_arabic``:
  Default source: ``Misraj/Sadeed_Tashkeela`` — gated, requires HF_TOKEN
  and access grant at
  https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela
  Fallback: ``community-datasets/tashkeela`` — GPLv2, open access, raw
  book text that needs heavier cleaning (handled by the data module).

For ``rababa_hebrew`` and ``secryst_thai_ipa`` the upstream sources are
not yet on HF as datasets — leave the manual env-var path intact in
``fetch_data.sh`` until they are.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "data" / "raw"

DATASETS = {
    "rababa_arabic": {
        "primary": {
            "repo_id": "Misraj/Sadeed_Tashkeela",
            "repo_type": "dataset",
            "files": [
                "data/train-00000-of-00003.parquet",
                "data/train-00001-of-00003.parquet",
                "data/train-00002-of-00003.parquet",
            ],
            "test_files": ["data/test-00000-of-00001.parquet"],
            "text_column": "text",
            "out_name": "tashkeela_plus_plus.txt",
            "split_lines": False,
            "note": (
                "Gated dataset. Visit "
                "https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela, "
                "log in, accept the terms, then export HF_TOKEN."
            ),
        },
        "fallback": {
            "repo_id": "community-datasets/tashkeela",
            "repo_type": "dataset",
            "files": None,
            "text_column": "text",
            "out_name": "tashkeela_plus_plus.txt",
            "split_lines": True,
            "note": (
                "Open-access raw corpus (GPLv2). Each row is a full book; "
                "we split on newlines and skip lines >1024 chars."
            ),
        },
    },
}


def fetch_task(
    task: str,
    out_dir: Path,
    max_samples: int | None,
    use_fallback: bool,
) -> Path:
    cfg = DATASETS.get(task)
    if cfg is None:
        raise SystemExit(
            f"No fetcher registered for task '{task}'. "
            f"Known: {sorted(DATASETS)}"
        )
    source = cfg["fallback"] if use_fallback else cfg["primary"]

    try:
        import importlib.util

        if importlib.util.find_spec("huggingface_hub") is None:
            raise ImportError("huggingface_hub not installed")
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: "
            "pip install -e '.[publish]'"
        ) from e

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source["out_name"]
    text_column = source["text_column"]

    files = source["files"]
    if files is None:
        files = _list_repo_files(source["repo_id"], source["repo_type"], token)

    count = _stream_parquet_to_text(
        files=files,
        repo_id=source["repo_id"],
        repo_type=source["repo_type"],
        token=token,
        text_column=text_column,
        out_path=out_path,
        max_samples=max_samples,
        split_lines=source.get("split_lines", False),
    )
    print(
        f"[{task}] wrote {count:,} lines -> {out_path} "
        f"({out_path.stat().st_size:,} bytes) "
        f"from {source['repo_id']}"
    )
    if count == 0:
        raise SystemExit(
            f"No lines written. Source note: {source.get('note', '')}"
        )
    return out_path


def _list_repo_files(repo_id: str, repo_type: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    return [f for f in files if f.endswith((".parquet", ".json", ".jsonl", ".txt"))]


def _stream_parquet_to_text(
    files: list[str],
    repo_id: str,
    repo_type: str,
    token: str | None,
    text_column: str,
    out_path: Path,
    max_samples: int | None,
    split_lines: bool = False,
    max_line_chars: int = 1024,
) -> int:
    """Stream the ``text_column`` of each parquet file to ``out_path``.

    Each row's text is written as one line (after whitespace collapse).
    If ``split_lines`` is set (raw book corpora), the row's text is split
    on embedded newlines first — one row may carry many verse-sized lines.
    Lines longer than ``max_line_chars`` are skipped (training chunks
    should be ~50-60 words; longer ones are typically misplits).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    written = 0
    tmp = out_path.with_suffix(".txt.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for fpath in files:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=fpath,
                repo_type=repo_type,
                token=token,
            )
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1024, columns=[text_column]):
                col = batch.column(text_column).to_pylist()
                for blob in col:
                    if not blob:
                        continue
                    chunks = blob.splitlines() if split_lines else [blob]
                    for raw in chunks:
                        line = " ".join(raw.split())
                        if not line or len(line) > max_line_chars:
                            continue
                        fp.write(line)
                        fp.write("\n")
                        written += 1
                        if max_samples is not None and written >= max_samples:
                            tmp.replace(out_path)
                            return written
    tmp.replace(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(DATASETS),
        help="Which task corpus to fetch.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of lines written (dev/CI mode).",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help=(
            "Use the open-access fallback dataset instead of the primary "
            "(useful when the primary is gated and no HF_TOKEN is set)."
        ),
    )
    args = parser.parse_args()

    try:
        fetch_task(
            task=args.task,
            out_dir=args.out_dir,
            max_samples=args.max_samples,
            use_fallback=args.fallback,
        )
    except SystemExit:
        raise
    except Exception as e:
        msg = str(e)
        if "GatedRepoError" in type(e).__name__ or "gated" in msg.lower():
            cfg = DATASETS[args.task]
            note = cfg["primary"].get("note", "")
            raise SystemExit(
                f"Gated dataset: {type(e).__name__}\n{note}\n"
                f"Or rerun with --fallback to use {cfg['fallback']['repo_id']}."
            ) from e
        raise SystemExit(f"Fetch failed: {type(e).__name__}: {msg}") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
