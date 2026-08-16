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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
