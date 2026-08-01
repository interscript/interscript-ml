"""Model layer abstractions.

Two roles per task: a *teacher* (large pretrained LLM, fine-tuned with
LoRA) and a *student* (small character-level transformer distilled from
the teacher). Both implement ``ModelModule`` so the trainer can swap
them without per-arch branching.

OCP: adding a new architecture = subclassing ``ModelModule`` and
registering via ``@register_model_module("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelKind(str, Enum):
    """Identifies a model's role. Drives trainer dispatch."""

    TEACHER = "teacher"
    STUDENT = "student"


@dataclass(frozen=True)
class ForwardOutput:
    """Output of ``ModelModule.forward``.

    ``logits`` shape: ``(batch, seq, vocab)``. ``hidden`` is the last
    layer's hidden states, used for distillation KL loss.
    """

    logits: Any  # torch.Tensor in training mode; Any keeps framework torch-optional.
    hidden: Any | None = None


@dataclass(frozen=True)
class GenerateOutput:
    """Output of ``ModelModule.generate``."""

    ids: Sequence[Sequence[int]]
    texts: Sequence[str]


class ModelModule(ABC):
    """Abstract model module.

    Subclasses provide ``forward``, ``generate``, ``save``, ``load``.
    The framework treats teacher/student polymorphically — adding a new
    architecture is one subclass, zero framework edits.
    """

    kind: ModelKind = ModelKind.STUDENT

    @property
    @abstractmethod
    def device(self) -> str:
        """The device the model lives on (e.g. ``"cuda"``, ``"cpu"``)."""

    @abstractmethod
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        labels: Any | None = None,
    ) -> ForwardOutput:
        """Forward pass. Returns logits + hidden states."""

    @abstractmethod
    def generate(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        max_new_tokens: int = 128,
    ) -> GenerateOutput:
        """Autoregressive generation for inference/teacher labeling."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist weights + tokenizer to ``path``."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore weights + tokenizer from ``path``."""

    @abstractmethod
    def parameters(self) -> Sequence[Any]:
        """Iterable of trainable parameters (for optimizer construction)."""
