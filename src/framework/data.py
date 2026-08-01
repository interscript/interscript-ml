"""Data layer abstractions.

The ``DataModule`` is the boundary between task-specific data fetching
(cleaning, encoding, augmentation) and the trainer. The trainer only
sees ``DataSplit`` objects with deterministic iteration.

OCP: adding a new data source = subclassing ``DataModule`` and
registering via ``@register_data_module("name")``. Framework code is
not modified.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from framework.config import DataConfig


@dataclass(frozen=True)
class Example:
    """One training example.

    Keeps the human-readable source/target alongside encoded IDs. The
    trainer uses ``input_ids``/``target_ids``; the evaluator uses
    ``source``/``target`` for metric computation.
    """

    source: str
    target: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Example.source cannot be empty")
        if not self.target:
            raise ValueError("Example.target cannot be empty")
        if not self.input_ids:
            raise ValueError("Example.input_ids cannot be empty")
        if not self.target_ids:
            raise ValueError("Example.target_ids cannot be empty")


@dataclass(frozen=True)
class DataSplit:
    """A frozen collection of ``Example`` objects.

    Frozen so a downstream component cannot accidentally mutate the
    dataset mid-epoch.
    """

    examples: tuple[Example, ...]

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


@dataclass(frozen=True)
class PreparedData:
    """Result of ``DataModule.prepare_data``."""

    train: DataSplit
    val: DataSplit
    vocab_size: int
    max_seq_len: int


class DataModule(ABC):
    """Abstract data module.

    Subclasses implement ``prepare_data`` (one-time fetch + clean +
    encode) and ``vocab_size``/``max_seq_len`` accessors. The framework
    calls only these — no data-format details leak upward.
    """

    def __init__(self, config: DataConfig, data_root: Path) -> None:
        self.config = config
        self.data_root = data_root
        self._prepared: PreparedData | None = None

    @property
    def prepared(self) -> PreparedData:
        if self._prepared is None:
            raise RuntimeError(
                "DataModule.prepare_data() must be called before access"
            )
        return self._prepared

    @abstractmethod
    def prepare_data(self) -> PreparedData:
        """Fetch + clean + encode. Idempotent: re-running is a no-op."""

    @abstractmethod
    def encode_source(self, text: str) -> tuple[int, ...]:
        """Encode raw source text to token IDs (for inference)."""

    @abstractmethod
    def decode_target(self, ids: Sequence[int]) -> str:
        """Decode target token IDs back to text (for inference output)."""
