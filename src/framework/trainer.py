"""Trainer abstractions.

``BaseTrainer`` implements the shared epoch loop, checkpointing, and
logging. Two concrete subclasses: ``FineTuneTrainer`` (for the teacher)
and ``DistillTrainer`` (for the student). All three are DRY: the loop
is written once, strategy-specific loss computation is delegated.

The trainer holds no knowledge of:
- Data source details (uses ``DataModule`` interface)
- Model architecture (uses ``ModelModule`` interface)
- Metric definitions (delegates to ``BaseEvaluator``)
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.config import TrainConfig
from framework.data import DataModule
from framework.model import ModelModule


@dataclass
class TrainingState:
    """Mutable training state. Tracked across epochs + steps."""

    epoch: int = 0
    step: int = 0
    best_loss: float = math.inf
    history: list[dict[str, float]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, **metrics: float) -> None:
        metrics["epoch"] = float(self.epoch)
        metrics["step"] = float(self.step)
        metrics["elapsed_s"] = time.time() - self.started_at
        self.history.append(metrics)


class BaseTrainer(ABC):
    """Shared trainer loop. Subclasses implement ``compute_loss``.

    Design:
    - ``fit`` runs the epoch loop, calls ``compute_loss`` per batch,
      tracks ``TrainingState``, periodically validates + checkpoints.
    - ``compute_loss`` is the only strategy-specific method. Fine-tune
      trainers compute CE loss against gold labels. Distill trainers
      mix CE + KL against teacher logits.
    """

    def __init__(
        self,
        config: TrainConfig,
        model: ModelModule,
        data: DataModule,
        out_dir: Path,
    ) -> None:
        self.config = config
        self.model = model
        self.data = data
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state = TrainingState()

    @abstractmethod
    def compute_loss(self, batch: Any) -> tuple[Any, dict[str, float]]:
        """Return ``(loss_tensor, scalar_metrics_dict)``.

        The trainer steps the optimizer on ``loss_tensor``. The dict is
        recorded into ``TrainingState.history``.
        """

    @abstractmethod
    def make_optimizer(self) -> Any:
        """Construct the optimizer over ``self.model.parameters()``."""

    def fit(self, max_steps: int | None = None) -> TrainingState:
        """Run ``config.epochs`` epochs. Subclasses normally don't override."""
        optimizer = self.make_optimizer()
        prepared = self.data.prepared
        steps_target = max_steps or (self.config.epochs * len(prepared.train))

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            for batch in self._iter_batches(prepared.train):
                self.state.step += 1
                loss, metrics = self.compute_loss(batch)
                self._step_optimizer(optimizer, loss)
                self._maybe_log(metrics)
                if self.state.step % self.config.save_every == 0:
                    self._checkpoint(tag=f"step-{self.state.step}")
                if self.state.step >= steps_target:
                    break

            val_metrics = self.validate()
            self.state.record(phase="val", **val_metrics)
            self._checkpoint(tag=f"epoch-{epoch}")
            if val_metrics.get("loss", math.inf) < self.state.best_loss:
                self.state.best_loss = val_metrics["loss"]
                self._checkpoint(tag="best")
            if self.state.step >= steps_target:
                break
        return self.state

    def validate(self) -> dict[str, float]:
        """Default: mean loss over val set. Subclasses may override."""
        prepared = self.data.prepared
        total = 0.0
        count = 0
        for batch in self._iter_batches(prepared.val):
            _, metrics = self.compute_loss(batch)
            total += metrics.get("loss", 0.0)
            count += 1
        return {"loss": total / max(count, 1)}

    def _iter_batches(self, split):
        """Yield batches of ``config.batch_size`` examples."""
        examples = list(split)
        for i in range(0, len(examples), self.config.batch_size):
            yield examples[i : i + self.config.batch_size]

    def _step_optimizer(self, optimizer: Any, loss: Any) -> None:
        loss.backward() if hasattr(loss, "backward") else None
        if self.config.grad_clip and hasattr(optimizer, "step"):
            self._clip_grads()
        if hasattr(optimizer, "step"):
            optimizer.step()
            optimizer.zero_grad()

    def _clip_grads(self) -> None:
        for p in self.model.parameters():
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.clamp_(-self.config.grad_clip, self.config.grad_clip)

    def _maybe_log(self, metrics: dict[str, float]) -> None:
        if self.state.step % self.config.log_every == 0:
            self.state.record(phase="train", **metrics)

    def _checkpoint(self, tag: str) -> None:
        self.model.save(str(self.out_dir / f"{tag}.ckpt"))


class FineTuneTrainer(BaseTrainer):
    """Teacher trainer: supervised CE loss on gold labels."""

    def compute_loss(self, batch):  # pragma: no cover - torch-bound
        raise NotImplementedError(
            "FineTuneTrainer requires torch; implement in tasks that use it"
        )

    def make_optimizer(self) -> Any:  # pragma: no cover - torch-bound
        raise NotImplementedError(
            "FineTuneTrainer requires torch; implement in tasks that use it"
        )


class DistillTrainer(BaseTrainer):
    """Student trainer: CE on gold + KL against teacher logits.

    The ``teacher`` is a frozen ``ModelModule`` (typically the result of
    ``FineTuneTrainer.fit``). ``alpha`` mixes CE and KL; standard value
    0.5.
    """

    def __init__(
        self,
        config: TrainConfig,
        model: ModelModule,
        teacher: ModelModule,
        data: DataModule,
        out_dir: Path,
    ) -> None:
        super().__init__(config, model, data, out_dir)
        self.teacher = teacher

    def compute_loss(self, batch):  # pragma: no cover - torch-bound
        raise NotImplementedError(
            "DistillTrainer requires torch; implement in tasks that use it"
        )

    def make_optimizer(self) -> Any:  # pragma: no cover - torch-bound
        raise NotImplementedError("DistillTrainer requires torch; implement in tasks that use it")
