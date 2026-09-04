"""interscript-sadeed-eval — score predictions on SadeedDiac-25 under
the campaign's windowed DER-CE convention.

    interscript-sadeed-eval score \\
        --preds final_preds.jsonl --data sadeed-diac-25.parquet \\
        [--key student] [--vs reference.jsonl] [--vs-key teacher]

Predictions are read as JSONL rows (any key; --key selects, default
"student" — the training harness's final_preds.jsonl writes
idx/src/teacher/student) or as plain text lines. The parquet carries
the benchmark's input/output columns; fetch SadeedDiac-25 from
https://huggingface.co/datasets/Misraj/SadeedDiac-25. Output is JSON
on stdout; --vs adds a paired-bootstrap delta (candidate minus
reference; positive = candidate worse)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_preds(path: Path, key: str) -> list[str]:
    preds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            preds.append(line)
            continue
        if not isinstance(row, dict) or key not in row:
            raise SystemExit(f"row is not a dict with key {key!r}: {line[:60]}")
        preds.append(row[key])
    return preds


def _load_gold(path: Path) -> list[str]:
    import pandas as pd

    table = pd.read_parquet(path)
    return table["output"].tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interscript-sadeed-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("score", help="score a predictions file")
    score.add_argument("--preds", type=Path, required=True)
    score.add_argument("--data", type=Path, required=True,
                       help="SadeedDiac-25 parquet (input/output columns)")
    score.add_argument("--key", default="student",
                       help="JSONL row key carrying the prediction")
    score.add_argument("--vs", type=Path, help="reference predictions file")
    score.add_argument("--vs-key", default="teacher")
    args = parser.parse_args(argv)

    from sadeedbench.bootstrap import bootstrap_delta
    from sadeedbench.scoring import per_item_der, score_predictions

    gts = _load_gold(args.data)
    preds = _read_preds(args.preds, args.key)
    result = score_predictions(preds, gts)
    if args.vs:
        ref = _read_preds(args.vs, args.vs_key)
        result["vs"] = bootstrap_delta(
            per_item_der(preds, gts), per_item_der(ref, gts)
        )
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
