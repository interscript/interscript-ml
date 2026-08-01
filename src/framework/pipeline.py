"""End-to-end training pipeline orchestrator.

Loads a task config, instantiates registered data/model/evaluator
classes, runs the trainer, then optionally runs ONNX export. The CLI
calls this — there is no task-specific orchestration anywhere in
framework code.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from framework.config import TaskConfig
from framework.data import DataModule
from framework.evaluator import BaseEvaluator, MetricSet
from framework.exporter import ExportResult, OnnxExporter
from framework.model import ModelModule
from framework.registry import (
    resolve_data_module,
    resolve_evaluator,
    resolve_model_module,
)
from framework.trainer import BaseTrainer

TrainerFactory = Callable[..., BaseTrainer]


@dataclass
class PipelineResult:
    """Everything the pipeline produced in one run."""

    task: str
    train_steps: int
    best_loss: float
    eval: MetricSet | None
    export: ExportResult | None


def _ensure_task_imported(task_name: str) -> None:
    """Import the task package so its @register_* decorators fire.

    Tasks are Python packages under ``src.tasks``. Importing them has
    the side-effect of registering their classes in the framework
    registries. This is the standard Python plugin pattern.
    """
    pkg = f"tasks.{task_name}"
    if importlib.util.find_spec(pkg) is None:
        raise ModuleNotFoundError(
            f"Task package '{pkg}' not found under src/tasks/"
        )
    importlib.import_module(pkg)


class TrainingPipeline:
    """Top-level orchestrator. Resolves classes by name from the config.

    Constructor takes explicit override hooks (``data_class``,
    ``model_class``, ``evaluator_class``, ``exporter``) so tests can
    inject doubles without touching the registry. Production callers go
    through ``from_config`` which uses the registry.
    """

    def __init__(
        self,
        config: TaskConfig,
        data_root: Path,
        out_root: Path,
        data_class: type[DataModule] | None = None,
        model_class: type[ModelModule] | None = None,
        evaluator_class: type[BaseEvaluator] | None = None,
        exporter: OnnxExporter | None = None,
        trainer_factory: TrainerFactory | None = None,
    ) -> None:
        self.config = config
        self.data_root = data_root
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._data_class = data_class
        self._model_class = model_class
        self._evaluator_class = evaluator_class
        self._exporter = exporter
        self._trainer_factory = trainer_factory

    @classmethod
    def from_config(
        cls,
        task_name: str,
        data_root: Path,
        out_root: Path,
        tasks_root: Path | None = None,
    ) -> TrainingPipeline:
        """Build a pipeline from ``src/tasks/<task_name>/config.yaml``."""
        from framework.config import load_task_config

        config = load_task_config(task_name, tasks_root=tasks_root)
        _ensure_task_imported(task_name)
        return cls(
            config=config,
            data_root=data_root,
            out_root=out_root,
            data_class=resolve_data_module(config.data.module),
            model_class=resolve_model_module(config.model.module),
            evaluator_class=resolve_evaluator(config.eval.module),
        )

    def build_data(self) -> DataModule:
        cls = self._data_class
        if cls is None:
            raise RuntimeError("Pipeline built without data_class")
        return cls(self.config.data, self.data_root)

    def build_model(self) -> ModelModule:
        cls = self._model_class
        if cls is None:
            raise RuntimeError("Pipeline built without model_class")
        return cls(self.config.model)

    def build_evaluator(self) -> BaseEvaluator:
        cls = self._evaluator_class
        if cls is None:
            raise RuntimeError("Pipeline built without evaluator_class")
        return cls()

    def build_exporter(self) -> OnnxExporter | None:
        if self._exporter is not None:
            return self._exporter
        # Default: try the torch exporter (works for any task whose model
        # implements export_to_onnx). Falls back to None if torch missing.
        try:
            from framework.torch_exporter import TorchOnnxExporter
            return TorchOnnxExporter(self.config.export)
        except ImportError:
            return None

    def run(
        self,
        max_steps: int | None = None,
        skip_export: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline: prepare → train → eval → export."""
        data = self.build_data()
        data.prepare_data()

        model = self.build_model()


        trainer = self._construct_trainer(model, data)
        state = trainer.fit(max_steps=max_steps)

        evaluator = self.build_evaluator()
        predictions = self._generate_predictions(model, data)
        gold = [example.target for example in data.prepared.val]
        metric = evaluator.evaluate(predictions, gold)

        export_result = None
        if not skip_export:
            exporter = self.build_exporter()
            if exporter is not None:
                out_path = self.out_root / f"{self.config.name}.onnx"
                export_result = exporter.export(model, out_path, verify_with=model)

        return PipelineResult(
            task=self.config.name,
            train_steps=state.step,
            best_loss=state.best_loss,
            eval=metric,
            export=export_result,
        )

    def _construct_trainer(self, model, data):
        """Build a trainer via the injected factory or default to StudentTrainer.

        StudentTrainer trains the student directly on gold labels (no
        teacher required) — works on CPU. The DistillTrainer path is
        used only when a teacher checkpoint exists; switch via
        ``trainer_factory`` injection in the constructor.
        """
        if self._trainer_factory is not None:
            return self._trainer_factory(
                config=self.config.train,
                model=model,
                data=data,
                out_dir=self.out_root / "checkpoints",
            )
        from framework.trainer import StudentTrainer

        return StudentTrainer(
            self.config.train,
            model,
            data,
            self.out_root / "checkpoints",
            device=self.config.model.device,
        )

    def _generate_predictions(self, model, data):
        """Generate predictions over the val split for evaluation.

        Models decide what tensor format they want. We pass the raw
        input_ids as a list-of-lists; torch-based models convert
        internally. Avoids importing torch at the framework layer
        (CPU-only test environments shouldn't fail here).
        """
        predictions: list[str] = []
        for example in data.prepared.val:
            try:
                output = model.generate([list(example.input_ids)])
                predictions.append(output.texts[0] if output.texts else "")
            except Exception:  # noqa: BLE001
                predictions.append("")
        return predictions
