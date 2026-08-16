"""Model index resolution + cached downloads (the dynamic-fetch layer).

Implements the models.yaml contract shared by the Ruby and TypeScript
runtimes: resolve an id, reuse a verified cache copy, or download +
sha256-verify + atomically install into the cache.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

DEFAULT_INDEX_URL = (
    "https://raw.githubusercontent.com/interscript/ml-models/main/models.yaml"
)
ENV_INDEX = "INTERSCRIPT_ML_INDEX"
ENV_CACHE = "INTERSCRIPT_ML_CACHE"


class RegistryError(ValueError):
    """The index cannot be fetched/parsed, or the id is unknown."""


@dataclass(frozen=True)
class IndexEntry:
    id: str
    filename: str
    url: str
    sha256: str
    size: int
    precision: str
    task: str


def cache_dir() -> Path:
    if os.environ.get(ENV_CACHE):
        return Path(os.environ[ENV_CACHE])
    return Path.home() / ".cache" / "interscript"


def load_index(index_url: str | None = None) -> dict[str, IndexEntry]:
    source = index_url or os.environ.get(ENV_INDEX) or DEFAULT_INDEX_URL
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise RegistryError("index must be a mapping with version: 1")
    entries: dict[str, IndexEntry] = {}
    for model_id, spec in raw.get("models", {}).items():
        entries[model_id] = IndexEntry(
            id=model_id,
            filename=spec["filename"],
            url=spec["url"],
            sha256=spec["sha256"],
            size=int(spec.get("size", 0)),
            precision=spec.get("precision", "fp32"),
            task=spec.get("task", ""),
        )
    return entries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(model_id: str, index_url: str | None = None) -> Path:
    """Return a verified local zip path for `model_id`, downloading and
    installing into the cache when needed. Never returns an unverified
    file: cache hits are re-verified against the index sha256."""
    entries = load_index(index_url)
    if model_id not in entries:
        raise RegistryError(
            f"unknown model id {model_id!r} (known: {sorted(entries)})"
        )
    entry = entries[model_id]
    target = cache_dir() / "models" / model_id / entry.filename
    if target.is_file() and _sha256_file(target) == entry.sha256:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(fd)
    downloaded = Path(tmp_name)
    if entry.url.startswith("file://"):
        source = Path(urlparse(entry.url).path)
        if not source.is_file():
            raise RegistryError(f"channel file missing: {source}")
        shutil.copyfile(source, downloaded)  # file:// is a mirror, not a move
    else:
        urllib.request.urlretrieve(entry.url, downloaded)
    try:
        actual = _sha256_file(downloaded)
        if actual != entry.sha256:
            raise RegistryError(
                f"downloaded {entry.filename} sha256 mismatch: got {actual}, "
                f"index says {entry.sha256}"
            )
        if downloaded != target:
            os.replace(downloaded, target)
    finally:
        if downloaded != target and downloaded.exists():
            downloaded.unlink()
    return target
