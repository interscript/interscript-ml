"""Canonical artifact naming: the index entry and release asset name
come from the zip's metadata, never the local staging path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytest.importorskip("yaml")

from publish_model import canonical_filename  # noqa: E402


def test_filename_from_metadata() -> None:
    meta = {"id": "ara-diac-layerdrop-1.0", "precision": "int4"}
    assert canonical_filename(meta) == "ara-diac-layerdrop-1.0-int4.zip"


def test_staging_path_cannot_leak() -> None:
    # a /tmp staging name like ld-int4.zip must never become the
    # published filename — the runtime resolver matches volume files
    # by the canonical pattern
    meta = {"id": "ara-diac-layerdrop-1.0", "precision": "fp32"}
    assert canonical_filename(meta) != "ld-fp32.zip"
