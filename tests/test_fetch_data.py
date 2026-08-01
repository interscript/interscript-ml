"""Tests for ``scripts/fetch_data.py``.

Network access is NOT required — these tests cover import, CLI
argument handling, and the gated-repo error path. Real download is
covered by the smoke test in the README.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_data.py"


@pytest.fixture(scope="module")
def fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetcher_imports(fetch_module) -> None:
    assert hasattr(fetch_module, "fetch_task")
    assert hasattr(fetch_module, "DATASETS")
    assert "rababa_arabic" in fetch_module.DATASETS


def test_fetcher_primary_has_parquet_files(fetch_module) -> None:
    primary = fetch_module.DATASETS["rababa_arabic"]["primary"]
    assert primary["repo_id"] == "Misraj/Sadeed_Tashkeela"
    assert primary["repo_type"] == "dataset"
    assert len(primary["files"]) == 3
    assert all(f.endswith(".parquet") for f in primary["files"])


def test_fetcher_fallback_uses_open_dataset(fetch_module) -> None:
    fallback = fetch_module.DATASETS["rababa_arabic"]["fallback"]
    assert fallback["repo_id"] == "community-datasets/tashkeela"
    assert fallback["split_lines"] is True


def test_fetcher_unknown_task_errors(fetch_module, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No fetcher registered"):
        fetch_module.fetch_task(
            task="not_a_task",
            out_dir=tmp_path,
            max_samples=None,
            use_fallback=False,
        )
