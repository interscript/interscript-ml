"""``python -m imf`` — validate, inspect, and pack IMF v1 model zips.

- ``validate <zip> [--strict]``  exit 0 iff the zip conforms
  (``--strict`` adds the release gate: metrics, parity, thresholds)
- ``info <zip>``                 print the parsed manifest
- ``pack``                       build a conforming zip from a legacy
  zip or a directory of graphs + a metadata YAML (sha256 computed here)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imf.pack import PackError, pack_zip
from imf.schema import ModelMetadata
from imf.validator import validate_zip


def _cmd_validate(args: argparse.Namespace) -> int:
    result = validate_zip(args.zip, strict=args.strict)
    for warning in result.warnings:
        print(f"warn: {warning}")
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    label = "strict " if args.strict else ""
    print(f"{args.zip}: {'OK' if result.ok else 'FAILED'} ({label}validation)")
    return 0 if result.ok else 1


def _cmd_info(args: argparse.Namespace) -> int:
    result = validate_zip(args.zip)
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    if result.metadata is None:
        return 1
    m = result.metadata
    print(f"id:            {m.id}")
    print(f"task:          {m.task} ({m.source_script} -> {m.target})")
    print(f"tokenizer:     {m.tokenizer}")
    print(f"decoder:       {m.decoder}")
    print(f"precision:     {m.precision}")
    print(f"opset:         {m.opset}")
    print(f"license:       {m.license}")
    print(f"trained_from:  {m.trained_from}")
    for metric in m.metrics:
        print(f"metric:        {metric.name} = {metric.value} [{metric.source}]")
    if m.parity is not None:
        print(f"parity:        cer_delta {m.parity.cer_delta}pp on {m.parity.samples} samples")
    else:
        print("parity:        (not measured)")
    for name, digest in sorted(m.sha256.items()):
        print(f"sha256:        {name} {digest}")
    return 0 if result.ok else 1


def _cmd_pack(args: argparse.Namespace) -> int:
    metadata = ModelMetadata.from_yaml(Path(args.metadata).read_text(encoding="utf-8"))
    readme = (
        Path(args.readme).read_text(encoding="utf-8") if args.readme else _default_readme(metadata)
    )
    try:
        out = pack_zip(args.source, metadata, readme, args.out)
    except PackError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    import sys

    import yaml

    from imf.metrics import check_against_metadata

    mapping = Path(args.mapping)
    entries = yaml.safe_load(mapping.read_text(encoding="utf-8"))
    entries = entries.get("models", entries)
    problems: list[str] = []
    for model_id, spec in entries.items():
        metadata = Path(spec.get("metadata")) if spec.get("metadata") else (
            Path("models") / model_id.rsplit("-", 1)[0] / f"{model_id}.metadata.yaml"
        )
        if not metadata.is_file():
            problems.append(f"{model_id}: metadata source not found at {metadata}")
            continue
        problems += check_against_metadata(model_id, metadata, mapping)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    label = "all models trace to RESULTS.md" if not problems else "MISMATCH"
    print(f"metrics provenance: {label} ({len(entries)} models)")
    return 0 if not problems else 1


def _default_readme(metadata: ModelMetadata) -> str:
    return (
        f"# {metadata.id}\n\n"
        f"{metadata.task} ({metadata.source_script} -> {metadata.target}), "
        f"{metadata.precision} precision, byte-level tokenizer, "
        f"decoder: {metadata.decoder}.\n\n"
        f"Trained from: {metadata.trained_from}\n"
        f"License: {metadata.license}\n\n"
        "IMF v1 artifact — see the interscript/ml-models docs/imf-v1.md spec.\n"
    )


def _load_pairs(path: Path) -> list[tuple[str, str]]:
    import json

    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            pairs.append(
                (
                    row.get("input", row.get("src", "")),
                    row.get("target", row.get("tgt", row.get("gold", ""))),
                )
            )
        else:
            pairs.append((row[0], row[1] if len(row) > 1 else ""))
    return pairs


def _cmd_parity(args: argparse.Namespace) -> int:
    from imf.export import load_byte_seq2seq
    from imf.parity import run_parity, write_parity

    model = load_byte_seq2seq(args.checkpoint)
    pairs = _load_pairs(args.test_data)
    if args.limit:
        pairs = pairs[: args.limit]
    report = run_parity(model, args.zip, pairs, max_len=args.max_len)
    print(
        f"samples={report.samples} cer_ref={report.cer_reference}pp "
        f"cer_onnx={report.cer_onnx}pp delta={report.cer_delta}pp "
        f"token_mismatches={report.token_mismatches}"
    )
    if not report.passed:
        print("error: parity gate FAILED", file=sys.stderr)
        return 1
    write_parity(args.zip, report)
    print(f"parity written into {args.zip} (strict validation passed)")
    return 0


def _cmd_golden(args: argparse.Namespace) -> int:
    import json

    from imf.parity import write_golden

    inputs: list[str] = []
    for line in args.inputs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            inputs.append(row.get("input", row.get("src", row.get("text", ""))))
        elif isinstance(row, str):
            inputs.append(row)
        else:
            inputs.append(row[0])
    out = write_golden(args.zip, inputs, args.out, max_len=args.max_len)
    print(f"wrote {len(inputs)} golden cases to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate a model.zip")
    p_validate.add_argument("zip", type=Path)
    p_validate.add_argument(
        "--strict", action="store_true", help="release gate: metrics + parity thresholds"
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_info = sub.add_parser("info", help="print the manifest of a model.zip")
    p_info.add_argument("zip", type=Path)
    p_info.set_defaults(func=_cmd_info)

    p_pack = sub.add_parser("pack", help="build a conforming model.zip")
    p_pack.add_argument(
        "--source", required=True, type=Path,
        help="directory of .onnx graphs, or a legacy zip to upgrade",
    )
    p_pack.add_argument(
        "--metadata", required=True, type=Path,
        help="metadata YAML (sha256 block is computed and overwritten)",
    )
    p_pack.add_argument("--readme", type=Path, help="README.md content for the zip")
    p_pack.add_argument("--out", required=True, type=Path)
    p_pack.set_defaults(func=_cmd_pack)

    p_parity = sub.add_parser(
        "parity", help="WO03 gate: ONNX vs torch reference, write parity into the zip"
    )
    p_parity.add_argument("zip", type=Path)
    p_parity.add_argument(
        "--checkpoint", required=True, type=Path, help="HF checkpoint dir (reference)"
    )
    p_parity.add_argument(
        "--test-data", required=True, type=Path,
        help="JSONL: {input, target} pairs (or [src, gold] arrays)",
    )
    p_parity.add_argument("--limit", type=int, help="cap sample count")
    p_parity.add_argument("--max-len", type=int, default=256)
    p_parity.set_defaults(func=_cmd_parity)

    p_golden = sub.add_parser(
        "golden", help="emit the cross-runtime golden JSONL from ONNX decode"
    )
    p_golden.add_argument("zip", type=Path)
    p_golden.add_argument(
        "--inputs", required=True, type=Path, help="JSONL of input strings"
    )
    p_golden.add_argument("--out", required=True, type=Path)
    p_golden.add_argument("--max-len", type=int, default=256)
    p_golden.set_defaults(func=_cmd_golden)

    p_metrics = sub.add_parser(
        "metrics", help="WO10: check every metadata metrics block against its RESULTS.md source"
    )
    p_metrics.add_argument("--mapping", type=Path, default=Path("models/metrics-sources.yaml"))
    p_metrics.set_defaults(func=_cmd_metrics)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
