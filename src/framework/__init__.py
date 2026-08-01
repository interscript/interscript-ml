"""Interscript ML training framework.

Shared abstractions used by every task (rababa, secryst, future). Each
module is MECE: data has no knowledge of model architecture, model has
no knowledge of trainer, trainer has no knowledge of evaluator. Adding
a new task is one directory under ``src/tasks/`` — zero edits here.
"""

from framework.config import TaskConfig, load_task_config
from framework.data import DataModule, DataSplit, Example
from framework.evaluator import BaseEvaluator, MetricSet
from framework.exporter import ExportResult, OnnxExporter
from framework.model import ModelKind, ModelModule
from framework.pipeline import TrainingPipeline
from framework.registry import (
    register_data_module,
    register_evaluator,
    register_model_module,
    resolve_data_module,
    resolve_evaluator,
    resolve_model_module,
)
from framework.trainer import BaseTrainer, TrainingState

__all__ = [
    "TaskConfig",
    "load_task_config",
    "DataModule",
    "DataSplit",
    "Example",
    "ModelModule",
    "ModelKind",
    "BaseTrainer",
    "TrainingState",
    "BaseEvaluator",
    "MetricSet",
    "OnnxExporter",
    "ExportResult",
    "TrainingPipeline",
    "register_data_module",
    "register_model_module",
    "register_evaluator",
    "resolve_data_module",
    "resolve_model_module",
    "resolve_evaluator",
]
