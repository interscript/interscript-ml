"""Tests for the dynamic-fetch layer (models.yaml resolution + cache)."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

from interscript_ml.registry import RegistryError, resolve
from tests_helpers import build_tiny_zip

import os  # noqa: E402


def _index_file(tmp_path: Path, zip_path: Path, sha256: str | None = None) -> Path:
    index = {
        "version": 1,
        "models": {
            "tiny-1.0": {
                "task": "translit",
                "precision": "fp32",
                "filename": zip_path.name,
                "url": f"file://{zip_path}",
                "sha256": sha256 or hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                "size": zip_path.stat().st_size,
            }
        },
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(index), encoding="utf-8")
    return path


def test_resolve_downloads_verifies_and_caches(tmp_path: Path) -> None:
    zip_path = build_tiny_zip(tmp_path / "channel" / "tiny.zip")
    index = _index_file(tmp_path, zip_path)
    cache = tmp_path / "cache"
    os.environ["INTERSCRIPT_ML_CACHE"] = str(cache)
    try:
        local = resolve("tiny-1.0", index_url=str(index))
        assert local == cache / "models" / "tiny-1.0" / "tiny.zip"
        assert local.is_file()
        # second resolve is a verified cache hit (channel dir removed)
        zip_path.unlink()
        assert resolve("tiny-1.0", index_url=str(index)) == local
    finally:
        os.environ.pop("INTERSCRIPT_ML_CACHE", None)


def test_resolve_rejects_bad_download(tmp_path: Path) -> None:
    zip_path = build_tiny_zip(tmp_path / "channel" / "tiny.zip")
    index = _index_file(tmp_path, zip_path, sha256="0" * 64)
    os.environ["INTERSCRIPT_ML_CACHE"] = str(tmp_path / "cache")
    try:
        with pytest.raises(RegistryError, match="sha256 mismatch"):
            resolve("tiny-1.0", index_url=str(index))
    finally:
        os.environ.pop("INTERSCRIPT_ML_CACHE", None)


def test_resolve_unknown_id(tmp_path: Path) -> None:
    index = tmp_path / "models.yaml"
    index.write_text(yaml.safe_dump({"version": 1, "models": {}}), encoding="utf-8")
    with pytest.raises(RegistryError, match="unknown model id"):
        resolve("nope-1.0", index_url=str(index))


def test_model_load_by_id(tmp_path: Path) -> None:
    zip_path = build_tiny_zip(tmp_path / "channel" / "tiny.zip")
    index = _index_file(tmp_path, zip_path)
    os.environ["INTERSCRIPT_ML_CACHE"] = str(tmp_path / "cache")
    try:
        from interscript_ml import Model

        model = Model.load("tiny-1.0", index_url=str(index))
        assert model.id == "tiny-1.0"
        assert isinstance(model.translate("he", max_len=4), str)
    finally:
        os.environ.pop("INTERSCRIPT_ML_CACHE", None)


def test_resolve_parts_assembles_and_verifies(tmp_path: Path) -> None:
    import hashlib

    zip_path = build_tiny_zip(tmp_path / "channel" / "tiny.zip")
    blob = zip_path.read_bytes()
    part_a, part_b = blob[: len(blob) // 2 + 3], blob[len(blob) // 2 + 3 :]
    channel = tmp_path / "channel"
    (channel / "tiny.zip.part-00").write_bytes(part_a)
    (channel / "tiny.zip.part-01").write_bytes(part_b)
    index = {
        "version": 1,
        "models": {
            "tiny-1.0": {
                "task": "translit",
                "precision": "fp32",
                "filename": "tiny.zip",
                "sha256": hashlib.sha256(blob).hexdigest(),
                "size": len(blob),
                "parts": [
                    {
                        "url": f"file://{channel / 'tiny.zip.part-00'}",
                        "sha256": hashlib.sha256(part_a).hexdigest(),
                        "size": len(part_a),
                    },
                    {
                        "url": f"file://{channel / 'tiny.zip.part-01'}",
                        "sha256": hashlib.sha256(part_b).hexdigest(),
                        "size": len(part_b),
                    },
                ],
            },
        },
    }
    index_path = tmp_path / "models.yaml"
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    cache = tmp_path / "cache"
    os.environ["INTERSCRIPT_ML_CACHE"] = str(cache)
    try:
        local = resolve("tiny-1.0", index_url=str(index_path))
        assert local == cache / "models" / "tiny-1.0" / "tiny.zip"
        assert local.read_bytes() == blob
        zip_path.unlink()
        (channel / "tiny.zip.part-00").unlink()
        assert resolve("tiny-1.0", index_url=str(index_path)) == local
    finally:
        os.environ.pop("INTERSCRIPT_ML_CACHE", None)


def test_resolve_parts_rejects_corrupt_part(tmp_path: Path) -> None:
    import hashlib

    zip_path = build_tiny_zip(tmp_path / "channel" / "tiny.zip")
    blob = zip_path.read_bytes()
    part_a, part_b = blob[:7], blob[7:]
    channel = tmp_path / "channel"
    (channel / "tiny.zip.part-00").write_bytes(part_a)
    (channel / "tiny.zip.part-01").write_bytes(part_b)
    index = {
        "version": 1,
        "models": {
            "tiny-1.0": {
                "filename": "tiny.zip",
                "sha256": hashlib.sha256(blob).hexdigest(),
                "parts": [
                    {
                        "url": f"file://{channel / 'tiny.zip.part-00'}",
                        "sha256": "0" * 64,
                        "size": len(part_a),
                    },
                    {
                        "url": f"file://{channel / 'tiny.zip.part-01'}",
                        "sha256": hashlib.sha256(part_b).hexdigest(),
                        "size": len(part_b),
                    },
                ],
            },
        },
    }
    index_path = tmp_path / "models.yaml"
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    os.environ["INTERSCRIPT_ML_CACHE"] = str(tmp_path / "cache")
    try:
        with pytest.raises(RegistryError, match="part 0"):
            resolve("tiny-1.0", index_url=str(index_path))
    finally:
        os.environ.pop("INTERSCRIPT_ML_CACHE", None)
