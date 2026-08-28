"""Index-driven model-id resolution: the models.yaml contract, shared
by every consumer. A model id resolves to the exact index filename
when the volume carries it, falling back to the precision convention
(``{id}-{precision}.zip``) and then ``{id}-fp32.zip`` — volume copies
have historically landed under convention names even when the release
artifact uses another name (heb-diac-1.1 ships as heb.zip but the
volume copy is heb-diac-1.1-fp32.zip).

Callers pass the volume listing; candidate names are built server-side
and matched by equality only — user input never touches path
construction (CWE-22).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_index(path: str | Path = "models.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["models"]


def resolve_zip_filename(model_id: str, index: dict, volume_files: list[str]) -> str:
    """Return the volume filename serving `model_id`.

    Raises KeyError when the id is unknown or no volume file matches.
    """
    entry = index.get(model_id)
    if entry is None:
        raise KeyError(model_id)

    candidates: list[str] = []
    filename = entry.get("filename")
    if filename:
        candidates.append(filename)
    precision = entry.get("precision")
    if precision:
        candidates.append(f"{model_id}-{precision}.zip")
    candidates.append(f"{model_id}-fp32.zip")

    available = set(volume_files)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(model_id)
