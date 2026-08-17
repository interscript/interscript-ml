"""WO10: RESULTS.md -> metadata metrics generator.

Every IMF zip's metrics block is generated from a RESULTS.md table —
never hand-written — and CI refuses to release a model whose metadata
disagrees with the documented protocol numbers.

Sources are pinned in models/metrics-sources.yaml (repo, ref, path,
anchor). Extraction is by table position (row label + optional column
header), not regexes over prose: the mapping says where a number lives,
the parser reads exactly that cell.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class MetricsError(ValueError):
    """The RESULTS.md source cannot yield the mapped metrics."""


@dataclass(frozen=True)
class TableSpec:
    row: str
    column: str | None = None  # None: first value cell in the row
    as_name: str = ""
    protocol: str | None = None  # override the source-level protocol


@dataclass(frozen=True)
class SourceSpec:
    repo: str
    ref: str
    path: str
    anchor: str
    protocol: str
    tables: tuple[TableSpec, ...]
    display_anchor: str = ""


def _slugify(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text).strip("-")


def _cell_to_value(cell: str) -> float:
    text = cell.replace("**", "").replace("%", "").replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise MetricsError(f"no numeric value in cell {cell!r}")
    return float(match.group())


def parse_tables(markdown: str, anchor: str) -> list[dict[str, Any]]:
    """All tables in the section whose heading slugifies to `anchor`."""
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("## "):
            if start is not None:
                end = index
                break
            if _slugify(line[3:]) == anchor:
                start = index
    else:
        end = len(lines) if start is not None else None
    if start is None:
        raise MetricsError(f"section anchor {anchor!r} not found")

    tables: list[dict[str, Any]] = []
    index = start
    while index < end:
        line = lines[index]
        if line.startswith("|") and index + 1 < end and set(lines[index + 1]) <= set("|-: "):
            header = [cell.strip() for cell in line.strip("|").split("|")]
            index += 2
            rows: list[dict[str, Any]] = []
            while index < end and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                rows.append({"label": cells[0], "cells": cells, "header": header})
                index += 1
            tables.append({"header": header, "rows": rows})
        else:
            index += 1
    if not tables:
        raise MetricsError(f"no tables under anchor {anchor!r}")
    return tables


def extract(
    markdown: str, anchor: str, specs: tuple[TableSpec, ...]
) -> list[dict[str, Any]]:
    tables = parse_tables(markdown, anchor)
    out: list[dict[str, Any]] = []
    for spec in specs:
        found = None
        for table in tables:
            for row in table["rows"]:
                if spec.row.lower() in row["label"].lower():
                    found = row
                    break
            if found:
                break
        if found is None:
            raise MetricsError(f"row {spec.row!r} not found under {anchor!r}")
        if spec.column is None:
            values = [
                _cell_to_value(cell)
                for cell in found["cells"][1:]
                if "%" in cell or re.search(r"\d", cell.replace("**", ""))
            ]
            if not values:
                raise MetricsError(f"no value cells in row {spec.row!r}")
            value = values[0]
        else:
            try:
                column_index = found["header"].index(spec.column)
            except ValueError as e:
                raise MetricsError(
                    f"column {spec.column!r} not in table header {found['header']}"
                ) from e
            value = _cell_to_value(found["cells"][column_index])
        out.append({"name": spec.as_name or spec.row, "value": value})
    return out


def load_source(source: SourceSpec, cache_dir: Path | None = None) -> str:
    url = (
        f"https://raw.githubusercontent.com/{source.repo}/{source.ref}/{source.path}"
    )
    if cache_dir is not None:
        cached = cache_dir / _cache_name(source)
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / _cache_name(source)).write_text(text, encoding="utf-8")
    return text


def _cache_name(source: SourceSpec) -> str:
    return f"{source.repo.replace('/', '_')}@{source.ref}_{source.path.replace('/', '_')}"




def generate_metrics(
    model_id: str,
    mapping_path: Path | str,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8"))
    entry = raw.get("models", raw).get(model_id)
    if entry is None:
        raise MetricsError(f"no metrics source mapped for {model_id!r}")
    source = SourceSpec(
        repo=entry["repo"],
        ref=entry["ref"],
        path=entry["path"],
        anchor=entry["anchor"],
        protocol=entry["protocol"],
        display_anchor=entry.get("display_anchor", ""),
        tables=tuple(
            TableSpec(
                row=t["row"],
                column=t.get("column"),
                as_name=t.get("as", ""),
                protocol=t.get("protocol"),
            )
            for t in entry["tables"]
        ),
    )
    markdown = load_source(source, cache_dir)
    extracted = extract(markdown, source.anchor, source.tables)
    source_ref = f"{source.path}#{source.display_anchor or source.anchor}"
    return [
        {
            "name": m["name"],
            "value": m["value"],
            "protocol": spec.protocol or source.protocol,
            "source": source_ref,
        }
        for m, spec in zip(extracted, source.tables, strict=True)
    ]


def check_against_metadata(
    model_id: str,
    metadata_path: Path | str,
    mapping_path: Path | str,
    cache_dir: Path | None = None,
) -> list[str]:
    """Diff generated metrics vs a metadata source file. Returns problems."""
    generated = generate_metrics(model_id, mapping_path, cache_dir)
    meta = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
    recorded = meta.get("metrics", [])
    problems: list[str] = []
    if [(m["name"], m["value"]) for m in generated] != [(m["name"], m["value"]) for m in recorded]:
        problems.append(
            f"{model_id}: metrics mismatch — generated "
            f"{[(m['name'], m['value']) for m in generated]} vs metadata "
            f"{[(m['name'], m['value']) for m in recorded]}"
        )
    return problems
