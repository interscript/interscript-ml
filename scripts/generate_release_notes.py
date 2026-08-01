"""Generate release notes for a task version.

Reads ``models/<task>/<task>-benchmarks.json`` (if present) and emits
a Markdown summary suitable for the GH Release body. The shape matches
``TODO.distribution/01-github-releases.md``.

Usage::

    python scripts/generate_release_notes.py --task rababa_arabic --version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TEMPLATE = """\
## {task} v{version}

{summary_line}

### Metrics

{metrics_table}

### Assets

{assets_table}

### How to use

```js
import {{ transliterateAsync }} from "interscript-ts"
const result = await transliterateAsync("{map_code}", "<input>")
```

```ruby
require "interscript"
Interscript.transliterate("{map_code}", "<input>")
```

### Direct download

```
https://github.com/interscript/ml-models/releases/download/{task}-v{version}/{task}.onnx
```

Full distribution plan: see ``TODO.distribution/`` in the source repo.
"""

MAP_CODES = {
    "rababa_arabic": "var-ara-Arab-Arab-rababa",
    "rababa_hebrew": "var-heb-Hebr-Hebr-rababa",
    "secryst_thai_ipa": "var-tha-Thai-Zsym-ipa",
}

SUMMARIES = {
    "rababa_arabic": "Arabic diacritization student model.",
    "rababa_hebrew": "Hebrew diacritization student model.",
    "secryst_thai_ipa": "Thai → IPA transliteration student model.",
}


def render(args: argparse.Namespace) -> str:
    task = args.task
    version = args.version
    summary = SUMMARIES.get(task, "Interscript ML model release.")
    map_code = MAP_CODES.get(task, "<map_code>")

    metrics_table = "_No benchmark file found for this release._"
    bench_path = Path(f"models/{task}/{task}-benchmarks.json")
    if not bench_path.is_file():
        bench_path = Path(f"{task}-benchmarks.json")
    if bench_path.is_file():
        try:
            data = json.loads(bench_path.read_text(encoding="utf-8"))
            metric_value = data.get("metrics", {}).get("der") or data.get(
                "metrics", {}
            ).get("per", "n/a")
            metric_name = "DER" if "der" in data.get("metrics", {}) else "PER"
            p95 = data.get("performance", {}).get("p95_ms", "n/a")
            metrics_table = (
                f"| Metric | Value |\n|---|---|\n"
                f"| {metric_name} | {metric_value} |\n"
                f"| p95 latency | {p95} ms/word |"
            )
        except Exception as exc:  # noqa: BLE001
            metrics_table = f"_Benchmark parse error: {exc}_"

    assets_table_lines = ["| File | Size |", "|---|---|"]
    assets_dir = Path(f"models/{task}")
    if not assets_dir.is_dir():
        assets_dir = Path(".")
    onnx_files = sorted(assets_dir.glob("*.onnx"))
    for f in onnx_files:
        assets_table_lines.append(f"| {f.name} | {f.stat().st_size / 1024 / 1024:.2f} MB |")
    if len(assets_table_lines) == 2:
        assets_table_lines.append("| _no assets staged yet_ | — |")
    assets_table = "\n".join(assets_table_lines)

    return TEMPLATE.format(
        task=task,
        version=version,
        summary_line=summary,
        metrics_table=metrics_table,
        assets_table=assets_table,
        map_code=map_code,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", default="RELEASE_NOTES.md")
    args = parser.parse_args(argv)
    notes = render(args)
    Path(args.out).write_text(notes, encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
